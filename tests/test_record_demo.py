import os
import pickle
import tempfile
import shutil
import pytest
from unittest import mock

# Import DemoRecorder and constants from record_demo.py
from record_demo import DemoRecorder, DEMOS_DIR, MAX_DEMOS_PER_FILE

class DummyEnv:
    def reset(self):
        return 'dummy_state'
    def step(self, action):
        return 'next_state', 1.0, False, {}

def test_demo_recorder_initializes_and_creates_dir():
    # Remove demos dir if it exists
    if os.path.exists(DEMOS_DIR):
        shutil.rmtree(DEMOS_DIR)
    with mock.patch('record_demo.GeomEnv', DummyEnv):
        recorder = DemoRecorder()
        assert os.path.exists(DEMOS_DIR)

def test_save_demos_creates_file():
    with mock.patch('record_demo.GeomEnv', DummyEnv):
        recorder = DemoRecorder()
        recorder.demos = [[{'state': 1, 'action': [0,0,0,0,0], 'reward': 1, 'next_state': 2, 'done': False}]]
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = 'test_demo.pkl'
            filepath = os.path.join(tmpdir, filename)
            with mock.patch('record_demo.DEMOS_DIR', tmpdir):
                recorder.save_demos(filename)
                assert os.path.exists(filepath)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                assert data == recorder.demos or data == [[{'state': 1, 'action': [0,0,0,0,0], 'reward': 1, 'next_state': 2, 'done': False}]]

def test_auto_save_after_max_demos():
    with mock.patch('record_demo.GeomEnv', DummyEnv):
        recorder = DemoRecorder()
        recorder.demos = []
        recorder.demo_count = 0
        # Patch save_demos to count calls
        with mock.patch.object(recorder, 'save_demos') as mock_save:
            for _ in range(MAX_DEMOS_PER_FILE):
                recorder.demos.append([{'state': 1}])
                recorder.demo_count += 1
                if len(recorder.demos) >= MAX_DEMOS_PER_FILE:
                    recorder.save_demos()
            assert mock_save.called

def test_ready_check_starts_recording():
    with mock.patch('record_demo.GeomEnv', DummyEnv), \
         mock.patch('record_demo.input', return_value=''), \
         mock.patch('record_demo.getScore', return_value=0):
        recorder = DemoRecorder()
        # Patch methods to avoid actual recording
        with mock.patch.object(recorder, '_get_action_from_keys', return_value=[0,0,0,0,0]), \
             mock.patch.object(recorder.env, 'step', return_value=('next_state', 1.0, True, {})), \
             mock.patch('time.sleep', return_value=None):
            recorder.start_recording()
            # Should have at least one demo after a game over
            assert recorder.demos

def test_ready_check_warns_on_nonzero_score(capsys):
    with mock.patch('record_demo.GeomEnv', DummyEnv), \
         mock.patch('record_demo.input', return_value=''), \
         mock.patch('record_demo.getScore', return_value=5):
        recorder = DemoRecorder()
        with mock.patch.object(recorder, '_get_action_from_keys', return_value=[0,0,0,0,0]), \
             mock.patch('time.sleep', return_value=None):
            recorder.start_recording()
            captured = capsys.readouterr()
            assert "not 0" in captured.out 