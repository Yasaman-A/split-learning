from tests.context import split-learning


def test_app(capsys, example_fixture):
    # pylint: disable=W0612,W0613
    split-learning.Main.run()
    captured = capsys.readouterr()

    assert "Hello World..." in captured.out
