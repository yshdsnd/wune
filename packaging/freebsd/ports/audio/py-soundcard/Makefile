PORTNAME=	soundcard
PORTVERSION=	0.4.6
CATEGORIES=	audio python
MASTER_SITES=	PYPI
PKGNAMEPREFIX=	${PYTHON_PKGNAMEPREFIX}

MAINTAINER=	yshd.snd@gmail.com
COMMENT=	Play and record audio without native CPython extensions
WWW=		https://github.com/bastibe/SoundCard

LICENSE=	BSD3CLAUSE

BUILD_DEPENDS=	${PYTHON_PKGNAMEPREFIX}cffi>0:devel/py-cffi@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}numpy>0:math/py-numpy@${PY_FLAVOR}
RUN_DEPENDS=	${PYTHON_PKGNAMEPREFIX}cffi>0:devel/py-cffi@${PY_FLAVOR} \
		${PYTHON_PKGNAMEPREFIX}numpy>0:math/py-numpy@${PY_FLAVOR} \
		pulseaudio:audio/pulseaudio

USES=		python
USE_PYTHON=	autoplist distutils

NO_ARCH=	yes

.include <bsd.port.mk>
