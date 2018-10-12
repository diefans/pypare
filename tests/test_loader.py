import aiohttp


class PyPiSession(aiohttp.ClientSession):
    api_base = 'https://pypi.org/pypi/{package}/json'


async def load_package_data(package):
    async with () as session:
        async with session.get(cls.api_base.format(package=package)) as r:
            package_data = await r.json()
            return package_data


async def test_load_package_data():
    package_data = await PyPiSession.load_package_data('implant')
    __import__('pdb').set_trace()      # XXX BREAKPOINT
    pass
