# Helper Makefile to build FreeBSD packages for Wune and dependencies
# Usage:
#   make package    - Build .pkg binary packages for py-soundcard and wune
#   make install    - Build and install py-soundcard and wune
#   make clean      - Clean build artifacts

PORTSDIR?=	/usr/ports

all: package

package: package-soundcard package-wune

package-soundcard:
	@echo "===> Building py-soundcard package..."
	${MAKE} -C ${.CURDIR}/ports/audio/py-soundcard PORTSDIR=${PORTSDIR} package

package-wune:
	@echo "===> Building wune package..."
	${MAKE} -C ${.CURDIR}/ports/audio/wune PORTSDIR=${PORTSDIR} package

install:
	${MAKE} -C ${.CURDIR}/ports/audio/py-soundcard PORTSDIR=${PORTSDIR} install clean
	${MAKE} -C ${.CURDIR}/ports/audio/wune PORTSDIR=${PORTSDIR} install clean

clean:
	${MAKE} -C ${.CURDIR}/ports/audio/py-soundcard PORTSDIR=${PORTSDIR} clean
	${MAKE} -C ${.CURDIR}/ports/audio/wune PORTSDIR=${PORTSDIR} clean
