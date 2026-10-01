#import <Foundation/Foundation.h>
#import <CoreAudio/CoreAudio.h>
#import <CoreAudio/CATapDescription.h>
#import <CoreAudio/AudioHardwareTapping.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RING_BUFFER_FRAMES 65536
#define CHANNELS 2

typedef struct WuneTapContext {
    AudioObjectID tapID;
    AudioDeviceID aggDeviceID;
    AudioDeviceIOProcID procID;
    uint32_t sampleRate;
    uint32_t channels;

    float ringBuffer[RING_BUFFER_FRAMES * CHANNELS];
    size_t writeIndex;
    size_t readIndex;
    size_t availableFrames;

    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool active;
} WuneTapContext;

static OSStatus tapAudioIOProc(
    AudioObjectID inDevice,
    const AudioTimeStamp* inNow,
    const AudioBufferList* inInputData,
    const AudioTimeStamp* inInputTime,
    AudioBufferList* outOutputData,
    const AudioTimeStamp* inOutputTime,
    void* inClientData)
{
    WuneTapContext *ctx = (WuneTapContext *)inClientData;
    if (!ctx || !ctx->active) return noErr;

    if (inInputData && inInputData->mNumberBuffers > 0) {
        const AudioBuffer *buf = &inInputData->mBuffers[0];
        const float *src = (const float *)buf->mData;
        uint32_t inFrames = buf->mDataByteSize / (sizeof(float) * buf->mNumberChannels);

        pthread_mutex_lock(&ctx->mutex);
        for (uint32_t i = 0; i < inFrames; i++) {
            if (ctx->availableFrames < RING_BUFFER_FRAMES) {
                float left = 0.0f;
                float right = 0.0f;
                if (buf->mNumberChannels >= 2) {
                    left = src[i * buf->mNumberChannels];
                    right = src[i * buf->mNumberChannels + 1];
                } else if (buf->mNumberChannels == 1) {
                    left = right = src[i];
                }
                ctx->ringBuffer[ctx->writeIndex * CHANNELS] = left;
                ctx->ringBuffer[ctx->writeIndex * CHANNELS + 1] = right;
                ctx->writeIndex = (ctx->writeIndex + 1) % RING_BUFFER_FRAMES;
                ctx->availableFrames++;
            }
        }
        pthread_cond_signal(&ctx->cond);
        pthread_mutex_unlock(&ctx->mutex);
    }
    return noErr;
}

WuneTapContext* wune_tap_create(uint32_t* out_sample_rate, uint32_t* out_channels) {
    @autoreleasepool {
        CATapDescription *desc = [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:@[]];
        if (!desc) return NULL;

        AudioObjectID tapID = 0;
        OSStatus status = AudioHardwareCreateProcessTap(desc, &tapID);
        if (status != noErr || tapID == 0) return NULL;

        // Get tap UID
        AudioObjectPropertyAddress tapUIDAddr = {
            .mSelector = kAudioTapPropertyUID,
            .mScope = kAudioObjectPropertyScopeGlobal,
            .mElement = kAudioObjectPropertyElementMain
        };
        CFStringRef tapUID = NULL;
        UInt32 size = sizeof(CFStringRef);
        status = AudioObjectGetPropertyData(tapID, &tapUIDAddr, 0, NULL, &size, &tapUID);
        if (status != noErr || !tapUID) {
            AudioHardwareDestroyProcessTap(tapID);
            return NULL;
        }

        // Get default output device UID for clock synchronization
        AudioDeviceID defaultOutput = 0;
        size = sizeof(AudioDeviceID);
        AudioObjectPropertyAddress defAddr = {
            .mSelector = kAudioHardwarePropertyDefaultOutputDevice,
            .mScope = kAudioObjectPropertyScopeGlobal,
            .mElement = kAudioObjectPropertyElementMain
        };
        AudioObjectGetPropertyData(kAudioObjectSystemObject, &defAddr, 0, NULL, &size, &defaultOutput);

        CFStringRef defUID = NULL;
        size = sizeof(CFStringRef);
        AudioObjectPropertyAddress uidAddr = {
            .mSelector = kAudioDevicePropertyDeviceUID,
            .mScope = kAudioObjectPropertyScopeGlobal,
            .mElement = kAudioObjectPropertyElementMain
        };
        AudioObjectGetPropertyData(defaultOutput, &uidAddr, 0, NULL, &size, &defUID);

        // Get nominal sample rate of default output device
        Float64 nominalSampleRate = 48000.0;
        size = sizeof(Float64);
        AudioObjectPropertyAddress srAddr = {
            .mSelector = kAudioDevicePropertyNominalSampleRate,
            .mScope = kAudioObjectPropertyScopeGlobal,
            .mElement = kAudioObjectPropertyElementMain
        };
        AudioObjectGetPropertyData(defaultOutput, &srAddr, 0, NULL, &size, &nominalSampleRate);

        // Create aggregate device
        NSString *aggUID = [NSString stringWithFormat:@"wune.tap.aggregate.%@", [[NSUUID UUID] UUIDString]];
        NSDictionary *subTapDict = @{ (__bridge NSString*)CFSTR("uid"): (__bridge NSString*)tapUID };
        NSDictionary *subDevDict = @{ (__bridge NSString*)CFSTR("uid"): (__bridge NSString*)defUID };

        NSDictionary *aggDict = @{
            (__bridge NSString*)CFSTR("name"): @"Wune System Audio Capture",
            (__bridge NSString*)CFSTR("uid"): aggUID,
            (__bridge NSString*)CFSTR("private"): @(1),
            (__bridge NSString*)CFSTR("stacked"): @(0),
            (__bridge NSString*)CFSTR("master"): (__bridge NSString*)defUID,
            (__bridge NSString*)CFSTR("subdevices"): @[ subDevDict ],
            (__bridge NSString*)CFSTR("taps"): @[ subTapDict ]
        };

        AudioDeviceID aggDevice = 0;
        status = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)aggDict, &aggDevice);
        if (tapUID) CFRelease(tapUID);
        if (defUID) CFRelease(defUID);

        if (status != noErr || aggDevice == 0) {
            AudioHardwareDestroyProcessTap(tapID);
            return NULL;
        }

        WuneTapContext *ctx = (WuneTapContext *)calloc(1, sizeof(WuneTapContext));
        if (!ctx) {
            AudioHardwareDestroyAggregateDevice(aggDevice);
            AudioHardwareDestroyProcessTap(tapID);
            return NULL;
        }

        ctx->tapID = tapID;
        ctx->aggDeviceID = aggDevice;
        ctx->sampleRate = (uint32_t)nominalSampleRate;
        ctx->channels = CHANNELS;
        ctx->active = true;
        pthread_mutex_init(&ctx->mutex, NULL);
        pthread_cond_init(&ctx->cond, NULL);

        status = AudioDeviceCreateIOProcID(aggDevice, tapAudioIOProc, ctx, &ctx->procID);
        if (status != noErr) {
            AudioHardwareDestroyAggregateDevice(aggDevice);
            AudioHardwareDestroyProcessTap(tapID);
            free(ctx);
            return NULL;
        }

        status = AudioDeviceStart(aggDevice, ctx->procID);
        if (status != noErr) {
            AudioDeviceDestroyIOProcID(aggDevice, ctx->procID);
            AudioHardwareDestroyAggregateDevice(aggDevice);
            AudioHardwareDestroyProcessTap(tapID);
            free(ctx);
            return NULL;
        }

        if (out_sample_rate) *out_sample_rate = ctx->sampleRate;
        if (out_channels) *out_channels = ctx->channels;
        return ctx;
    }
}

uint32_t wune_tap_read(WuneTapContext* ctx, float* buffer, uint32_t num_frames) {
    if (!ctx || !buffer || num_frames == 0) return 0;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    // 50ms timeout
    ts.tv_nsec += 50000000;
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000;
    }

    pthread_mutex_lock(&ctx->mutex);
    while (ctx->availableFrames < num_frames && ctx->active) {
        int r = pthread_cond_timedwait(&ctx->cond, &ctx->mutex, &ts);
        if (r != 0) break; // timeout
    }

    uint32_t framesToRead = (uint32_t)(ctx->availableFrames < num_frames ? ctx->availableFrames : num_frames);
    for (uint32_t i = 0; i < framesToRead; i++) {
        buffer[i * CHANNELS] = ctx->ringBuffer[ctx->readIndex * CHANNELS];
        buffer[i * CHANNELS + 1] = ctx->ringBuffer[ctx->readIndex * CHANNELS + 1];
        ctx->readIndex = (ctx->readIndex + 1) % RING_BUFFER_FRAMES;
    }
    ctx->availableFrames -= framesToRead;
    pthread_mutex_unlock(&ctx->mutex);

    // If frames were missing due to timeout (silence), fill remainder with zero
    if (framesToRead < num_frames) {
        memset(buffer + (framesToRead * CHANNELS), 0, (num_frames - framesToRead) * CHANNELS * sizeof(float));
    }
    return num_frames;
}

void wune_tap_destroy(WuneTapContext* ctx) {
    if (!ctx) return;
    ctx->active = false;

    pthread_mutex_lock(&ctx->mutex);
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);

    if (ctx->aggDeviceID && ctx->procID) {
        AudioDeviceStop(ctx->aggDeviceID, ctx->procID);
        AudioDeviceDestroyIOProcID(ctx->aggDeviceID, ctx->procID);
    }
    if (ctx->aggDeviceID) {
        AudioHardwareDestroyAggregateDevice(ctx->aggDeviceID);
    }
    if (ctx->tapID) {
        AudioHardwareDestroyProcessTap(ctx->tapID);
    }

    pthread_mutex_destroy(&ctx->mutex);
    pthread_cond_destroy(&ctx->cond);
    free(ctx);
}
