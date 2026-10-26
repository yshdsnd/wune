# Helper Makefile to build FreeBSD packages for Wune and dependencies
# Usage:
#   make package    - Build .pkg binary packages for py-soundcard and wune
#   make install    - Build and install py-soundcard and wune
#   make clean      - Clean build artifacts

PORTSDIR?=	/usr/ports
DISTDIR?=	${.CURDIR}/distfiles
PACKAGES?=	${.CURDIR}/packages

# If DISTVERSION is specified for Wune (or via WUNE_VERSION), make sure py-soundcard
# does not inherit it as an override, preserving soundcard's own PORTVERSION (0.4.6).
SOUNDCARD_DISTVERSION?=	0.4.6

WUNE_DISTVERSION_ARG=
.if defined(WUNE_VERSION)
WUNE_DISTVERSION_ARG=	DISTVERSION=${WUNE_VERSION}
.elif defined(DISTVERSION)
WUNE_DISTVERSION_ARG=	DISTVERSION=${DISTVERSION}
.endif

all: package

package: package-soundcard package-wune
	@mkdir -p ${PACKAGES}
	@find ${.CURDIR}/ports -name "*.pkg" -exec cp -f {} ${PACKAGES}/ \;
	@echo "===> Packages built successfully in ${PACKAGES}:"
	@ls -lh ${PACKAGES}/*.pkg

package-soundcard:
	@echo "===> Building py-soundcard package..."
	${MAKE} -C ${.CURDIR}/ports/audio/py-soundcard PORTSDIR=${PORTSDIR} DISTDIR=${DISTDIR} PACKAGES=${PACKAGES} DISTVERSION=${SOUNDCARD_DISTVERSION} package

package-wune:
	@echo "===> Building wune package..."
	@if pkg info -e "py*-soundcard*" >/dev/null 2>&1; then \
		${MAKE} -C ${.CURDIR}/ports/audio/wune ${WUNE_DISTVERSION_ARG} PORTSDIR=${PORTSDIR} DISTDIR=${DISTDIR} PACKAGES=${PACKAGES} package; \
	else \
		echo "===> py-soundcard is not yet installed on system; building wune with NO_DEPENDS=yes..."; \
		${MAKE} -C ${.CURDIR}/ports/audio/wune ${WUNE_DISTVERSION_ARG} PORTSDIR=${PORTSDIR} DISTDIR=${DISTDIR} PACKAGES=${PACKAGES} NO_DEPENDS=yes package; \
	fi

install:
	${MAKE} -C ${.CURDIR}/ports/audio/py-soundcard PORTSDIR=${PORTSDIR} DISTDIR=${DISTDIR} PACKAGES=${PACKAGES} DISTVERSION=${SOUNDCARD_DISTVERSION} install clean
	${MAKE} -C ${.CURDIR}/ports/audio/wune ${WUNE_DISTVERSION_ARG} PORTSDIR=${PORTSDIR} DISTDIR=${DISTDIR} PACKAGES=${PACKAGES} install clean

clean:
	${MAKE} -C ${.CURDIR}/ports/audio/py-soundcard PORTSDIR=${PORTSDIR} DISTVERSION=${SOUNDCARD_DISTVERSION} clean
	${MAKE} -C ${.CURDIR}/ports/audio/wune ${WUNE_DISTVERSION_ARG} PORTSDIR=${PORTSDIR} clean

distclean: clean
	rm -rf ${DISTDIR} ${PACKAGES}
