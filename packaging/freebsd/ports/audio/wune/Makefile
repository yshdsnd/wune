PORTNAME=	wune
DISTVERSION=	1.2.1.s20261024
CATEGORIES=	audio

MAINTAINER=	yshd.snd@gmail.com
COMMENT=	80s-90s mini-hifi-inspired LED audio spectrum visualizer
WWW=		https://github.com/yshdsnd/wune

LICENSE=	BSD2CLAUSE
LICENSE_FILE=	${WRKSRC}/LICENSE

# Hybrid Mode: If building inside the repository checkout, use local sources directly
# without fetching upstream distfiles or checking checksums.
# Otherwise, fetch upstream snapshot from GitHub.
.if exists(${.CURDIR}/../../../../../wune/__init__.py) && !defined(USE_UPSTREAM_DIST)
WRKSRC=		${.CURDIR}/../../../../..
NO_CHECKSUM=	yes
DISTFILES=
do-extract:
	@${DO_NADA}
.else
USE_GITHUB=	yes
GH_ACCOUNT=	yshdsnd
GH_PROJECT=	wune
GH_TAGNAME=	9e49cd3
.endif

RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}cffi>0:devel/py-cffi@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}numpy>0:math/py-numpy@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}game>=2.6.1:devel/py-game@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}soundcard>=0.4.6:audio/py-soundcard@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}tkinter>0:x11-toolkits/py-tkinter@${PY_FLAVOR} \
		pulseaudio:audio/pulseaudio

USES=		desktop-file-utils python
USE_PYTHON=	autoplist distutils

OPTIONS_DEFINE=		MPRIS PAVUCONTROL
OPTIONS_DEFAULT=	MPRIS PAVUCONTROL

MPRIS_DESC=		Now Playing track display via playerctl
MPRIS_RUN_DEPENDS=	playerctl:multimedia/playerctl

PAVUCONTROL_DESC=	Audio routing and volume control via pavucontrol
PAVUCONTROL_RUN_DEPENDS=pavucontrol:audio/pavucontrol

NO_ARCH=	yes

PLIST_FILES=	share/applications/wune.desktop \
		share/pixmaps/wune.png


post-install:
	${MKDIR} ${STAGEDIR}${PREFIX}/share/applications
	${INSTALL_DATA} ${FILESDIR}/wune.desktop ${STAGEDIR}${PREFIX}/share/applications/
	${MKDIR} ${STAGEDIR}${PREFIX}/share/pixmaps
	${INSTALL_DATA} ${WRKSRC}/wune/assets/wune.png ${STAGEDIR}${PREFIX}/share/pixmaps/wune.png

.include <bsd.port.mk>
