def test_project_from_dict(test_project_data):
    from pypare import pypi
    import attr
    import json

    dct = json.loads(test_project_data)
    project = pypi.Project.from_dict(dct)
    project_dict = attr.asdict(project)
    assert dct == project_dict
