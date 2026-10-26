PORTNAME=	wune
DISTVERSION=	1.3.0
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

OPTIONS_DEFINE=		DOCS MPRIS
OPTIONS_DEFAULT=	DOCS MPRIS

DOCSDIR=		${PREFIX}/share/doc/${PORTNAME}-${DISTVERSION}
PORTDOCS=		README.md README.en.md INSTALL_FREEBSD.md INSTALL_FREEBSD.en.md

MPRIS_DESC=		Now Playing track display via playerctl
MPRIS_RUN_DEPENDS=	playerctl:multimedia/playerctl

NO_ARCH=	yes

PLIST_FILES=	share/applications/wune.desktop \
		share/pixmaps/wune.png \
		%%PYTHON_SITELIBDIR%%/wune/build-info.json


post-install:
	${MKDIR} ${STAGEDIR}${PREFIX}/share/applications
	${INSTALL_DATA} ${FILESDIR}/wune.desktop ${STAGEDIR}${PREFIX}/share/applications/
	${MKDIR} ${STAGEDIR}${PREFIX}/share/pixmaps
	${INSTALL_DATA} ${WRKSRC}/wune/assets/wune.png ${STAGEDIR}${PREFIX}/share/pixmaps/wune.png
	@${PRINTF} '{\n  "release_version": "%s"\n}\n' "${DISTVERSION}" > \
		${STAGEDIR}${PYTHONPREFIX_SITELIBDIR}/wune/build-info.json

post-install-DOCS-on:
	${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${WRKSRC}/README.md ${STAGEDIR}${DOCSDIR}/
	${INSTALL_DATA} ${WRKSRC}/README.en.md ${STAGEDIR}${DOCSDIR}/
	${INSTALL_DATA} ${WRKSRC}/INSTALL_FREEBSD.md ${STAGEDIR}${DOCSDIR}/
	${INSTALL_DATA} ${WRKSRC}/INSTALL_FREEBSD.en.md ${STAGEDIR}${DOCSDIR}/

.include <bsd.port.mk>
