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
#define MAX_STACK_FRAMES 2048

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

static void ring_write_frames(WuneTapContext *ctx, const float *src, uint32_t frames) {
    if (!ctx || !src || frames == 0) return;

    if (frames > RING_BUFFER_FRAMES) {
        src += (frames - RING_BUFFER_FRAMES) * CHANNELS;
        frames = RING_BUFFER_FRAMES;
    }

    // Overwrite behavior: if full, advance readIndex to drop oldest frames
    if (ctx->availableFrames + frames > RING_BUFFER_FRAMES) {
        uint32_t overflow = (uint32_t)((ctx->availableFrames + frames) - RING_BUFFER_FRAMES);
        ctx->readIndex = (ctx->readIndex + overflow) % RING_BUFFER_FRAMES;
        ctx->availableFrames = RING_BUFFER_FRAMES - frames;
    }

    uint32_t part1 = (uint32_t)(RING_BUFFER_FRAMES - ctx->writeIndex);
    if (part1 > frames) part1 = frames;
    uint32_t part2 = frames - part1;

    memcpy(&ctx->ringBuffer[ctx->writeIndex * CHANNELS], src, part1 * CHANNELS * sizeof(float));
    if (part2 > 0) {
        memcpy(&ctx->ringBuffer[0], src + (part1 * CHANNELS), part2 * CHANNELS * sizeof(float));
        ctx->writeIndex = part2;
    } else {
        ctx->writeIndex = (ctx->writeIndex + part1) % RING_BUFFER_FRAMES;
    }
    ctx->availableFrames += frames;
}

static uint32_t ring_read_frames(WuneTapContext *ctx, float *dst, uint32_t frames) {
    if (!ctx || !dst || frames == 0) return 0;
    if (frames > ctx->availableFrames) {
        frames = (uint32_t)ctx->availableFrames;
    }
    if (frames == 0) return 0;

    uint32_t part1 = (uint32_t)(RING_BUFFER_FRAMES - ctx->readIndex);
    if (part1 > frames) part1 = frames;
    uint32_t part2 = frames - part1;

    memcpy(dst, &ctx->ringBuffer[ctx->readIndex * CHANNELS], part1 * CHANNELS * sizeof(float));
    if (part2 > 0) {
        memcpy(dst + (part1 * CHANNELS), &ctx->ringBuffer[0], part2 * CHANNELS * sizeof(float));
        ctx->readIndex = part2;
    } else {
        ctx->readIndex = (ctx->readIndex + part1) % RING_BUFFER_FRAMES;
    }
    ctx->availableFrames -= frames;
    return frames;
}

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
    if (!inInputData || inInputData->mNumberBuffers == 0) return noErr;

    uint32_t numBuffers = inInputData->mNumberBuffers;
    const AudioBuffer *b0 = &inInputData->mBuffers[0];
    if (!b0->mData || b0->mDataByteSize == 0) return noErr;

    if (numBuffers >= 2) {
        // Non-interleaved buffers: Buffer 0 = Left, Buffer 1 = Right
        const AudioBuffer *b1 = &inInputData->mBuffers[1];
        if (!b1->mData || b1->mDataByteSize == 0) return noErr;

        const float *srcL = (const float *)b0->mData;
        const float *srcR = (const float *)b1->mData;
        uint32_t framesL = b0->mDataByteSize / (sizeof(float) * (b0->mNumberChannels > 0 ? b0->mNumberChannels : 1));
        uint32_t framesR = b1->mDataByteSize / (sizeof(float) * (b1->mNumberChannels > 0 ? b1->mNumberChannels : 1));
        uint32_t inFrames = framesL < framesR ? framesL : framesR;

        uint32_t offset = 0;
        while (offset < inFrames) {
            uint32_t chunk = inFrames - offset;
            if (chunk > MAX_STACK_FRAMES) chunk = MAX_STACK_FRAMES;

            float temp[MAX_STACK_FRAMES * CHANNELS];
            for (uint32_t i = 0; i < chunk; i++) {
                temp[i * CHANNELS] = srcL[offset + i];
                temp[i * CHANNELS + 1] = srcR[offset + i];
            }

            pthread_mutex_lock(&ctx->mutex);
            ring_write_frames(ctx, temp, chunk);
            pthread_cond_signal(&ctx->cond);
            pthread_mutex_unlock(&ctx->mutex);

            offset += chunk;
        }
    } else {
        // Single buffer: Interleaved stereo (or mono / multichannel)
        const float *src = (const float *)b0->mData;
        uint32_t ch = b0->mNumberChannels > 0 ? b0->mNumberChannels : 1;
        uint32_t inFrames = b0->mDataByteSize / (sizeof(float) * ch);

        if (ch == CHANNELS) {
            // Standard interleaved stereo: direct fast bulk write
            pthread_mutex_lock(&ctx->mutex);
            ring_write_frames(ctx, src, inFrames);
            pthread_cond_signal(&ctx->cond);
            pthread_mutex_unlock(&ctx->mutex);
        } else {
            // Mono or multichannel: interleave into temp buffer
            uint32_t offset = 0;
            while (offset < inFrames) {
                uint32_t chunk = inFrames - offset;
                if (chunk > MAX_STACK_FRAMES) chunk = MAX_STACK_FRAMES;

                float temp[MAX_STACK_FRAMES * CHANNELS];
                for (uint32_t i = 0; i < chunk; i++) {
                    if (ch >= 2) {
                        temp[i * CHANNELS] = src[(offset + i) * ch];
                        temp[i * CHANNELS + 1] = src[(offset + i) * ch + 1];
                    } else {
                        temp[i * CHANNELS] = temp[i * CHANNELS + 1] = src[offset + i];
                    }
                }

                pthread_mutex_lock(&ctx->mutex);
                ring_write_frames(ctx, temp, chunk);
                pthread_cond_signal(&ctx->cond);
                pthread_mutex_unlock(&ctx->mutex);

                offset += chunk;
            }
        }
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
        status = AudioObjectGetPropertyData(kAudioObjectSystemObject, &defAddr, 0, NULL, &size, &defaultOutput);
        if (status != noErr || defaultOutput == 0) {
            CFRelease(tapUID);
            AudioHardwareDestroyProcessTap(tapID);
            return NULL;
        }

        CFStringRef defUID = NULL;
        size = sizeof(CFStringRef);
        AudioObjectPropertyAddress uidAddr = {
            .mSelector = kAudioDevicePropertyDeviceUID,
            .mScope = kAudioObjectPropertyScopeGlobal,
            .mElement = kAudioObjectPropertyElementMain
        };
        status = AudioObjectGetPropertyData(defaultOutput, &uidAddr, 0, NULL, &size, &defUID);
        if (status != noErr || !defUID) {
            if (defUID) CFRelease(defUID);
            CFRelease(tapUID);
            AudioHardwareDestroyProcessTap(tapID);
            return NULL;
        }

        // Get nominal sample rate of default output device
        Float64 nominalSampleRate = 48000.0;
        size = sizeof(Float64);
        AudioObjectPropertyAddress srAddr = {
            .mSelector = kAudioDevicePropertyNominalSampleRate,
            .mScope = kAudioObjectPropertyScopeGlobal,
            .mElement = kAudioObjectPropertyElementMain
        };
        status = AudioObjectGetPropertyData(defaultOutput, &srAddr, 0, NULL, &size, &nominalSampleRate);
        if (status != noErr || !isfinite(nominalSampleRate) || nominalSampleRate <= 0) {
            CFRelease(defUID);
            CFRelease(tapUID);
            AudioHardwareDestroyProcessTap(tapID);
            return NULL;
        }

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
            pthread_cond_destroy(&ctx->cond);
            pthread_mutex_destroy(&ctx->mutex);
            AudioHardwareDestroyAggregateDevice(aggDevice);
            AudioHardwareDestroyProcessTap(tapID);
            free(ctx);
            return NULL;
        }

        status = AudioDeviceStart(aggDevice, ctx->procID);
        if (status != noErr) {
            AudioDeviceDestroyIOProcID(aggDevice, ctx->procID);
            pthread_cond_destroy(&ctx->cond);
            pthread_mutex_destroy(&ctx->mutex);
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

uint32_t wune_tap_read(WuneTapContext* ctx, float* buffer, uint32_t num_frames, uint32_t timeout_ms) {
    if (!ctx || !buffer || num_frames == 0) return 0;

    if (timeout_ms == 0) {
        uint32_t sr = ctx->sampleRate ? ctx->sampleRate : 48000;
        timeout_ms = (uint32_t)((uint64_t)num_frames * 1500 / sr) + 50;
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t nsec = (uint64_t)ts.tv_nsec + ((uint64_t)timeout_ms * 1000000ULL);
    ts.tv_sec += nsec / 1000000000ULL;
    ts.tv_nsec = nsec % 1000000000ULL;

    pthread_mutex_lock(&ctx->mutex);
    while (ctx->availableFrames < num_frames && ctx->active) {
        int r = pthread_cond_timedwait(&ctx->cond, &ctx->mutex, &ts);
        if (r != 0) break; // timeout
    }

    uint32_t framesRead = ring_read_frames(ctx, buffer, num_frames);
    pthread_mutex_unlock(&ctx->mutex);

    return framesRead;
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
