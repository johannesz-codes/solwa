"""Cloud-free lifecycle tests: no network, keys, GPU, or paid resources."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ci.nebius import runner as r

REPO = 'johannesz-codes/solwa'
NAME = r.run_name(REPO, 123, 1)


def resource(kind, name=NAME, repo=REPO, now=0):
    return {'metadata': {'id': kind + '-id-' + r.owner_label(repo) + '-' + name, 'name': name,
                         'labels': r.resource_labels(repo, name, now)}}


class FakeCloud:
    project = 'project-test'

    def __init__(self, instances=(), disks=(), fail_delete=False, fail_create=False, states=()):
        self.data = {'instance': list(instances), 'disk': list(disks)}
        self.states = list(states)  # VM state reported on each successive list
        self.calls = []
        self.ops = []  # Nebius operations on the VM, newest first
        self.deleted = []
        self.fail_delete = fail_delete
        self.fail_create = fail_create
        self.payload = None

    def operations(self, instance_id):
        return list(self.ops)

    def resources(self, kind):
        if kind == 'instance' and self.states:
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            for vm in self.data['instance']:
                vm['status'] = {'state': state, 'reconciling': getattr(self, 'reconciling', False)}
        return iter(list(self.data[kind]))

    def delete(self, kind, resource_id):
        if self.fail_delete:
            raise r.CIError('Deletion failed')
        self.deleted.append((kind, resource_id))
        self.data[kind] = [x for x in self.data[kind] if x['metadata']['id'] != resource_id]

    def call(self, *args, payload=None, parse=True, timeout=180):
        self.calls.append(args[:3])
        if args[:3] != ('compute', 'instance', 'create'):
            return 'console log line' if 'logs' in args else ''
        self.payload = payload
        # Model cloud acceptance followed by a lost response.
        self.data['instance'].append({'metadata': {**payload['metadata'], 'id': 'vm-created'}})
        if self.fail_create:
            raise r.CIError('Create response lost')
        return {}


class FakeGitHub:
    repo = REPO

    def __init__(self, online=False):
        self.online = online
        self.entries = []
        self.deleted = []

    def resolve(self, ref):
        return 'a' * 40

    def runners(self):
        return iter(list(self.entries))

    def request(self, method, path, data=None, missing_ok=False):
        if method == 'POST':
            self.entries.append({'id': 123, 'name': data['name'], 'busy': False,
                                 'status': 'online' if self.online else 'offline'})
            return {'encoded_jit_config': 'JIT-ONE-TIME-SECRET'}
        if method == 'DELETE':
            self.deleted.append(path)
            self.entries = []
            return {}
        if path.startswith('actions/runs/'):
            return {'updated_at': '1970-01-01T00:00:00Z'}
        raise AssertionError((method, path))


class LifecycleTests(unittest.TestCase):
    def test_resource_spec_is_single_gpu_managed_disk_without_cloud_credentials(self):
        spec = r.instance_spec('project', 'subnet', 'gpu-l40s-a', '1gpu-8vcpu-32gb',
                               'ubuntu24.04-cuda13.0', REPO, NAME, 'short-lived-jit', 10)
        self.assertNotIn('service_account_id', spec['spec'])
        self.assertEqual(spec['spec']['boot_disk']['managed_disk']['spec']['size_gibibytes'], 80)
        self.assertEqual(spec['metadata']['labels']['expires-at'], '9010')
        config = json.loads(spec['spec']['cloud_init_user_data'].split('\n', 1)[1])
        self.assertFalse(config['users'][0]['sudo'])
        self.assertNotIn('NEBIUS_CREDENTIALS_JSON', json.dumps(config))
        self.assertNotIn('GH_RUNNER_TOKEN', json.dumps(config))

    def test_ref_is_never_interpolated_into_bootstrap(self):
        config = r.cloud_init(REPO, NAME, "'; $(touch /tmp/unsafe)\n")
        data = json.loads(config.split('\n', 1)[1])
        self.assertEqual(data['write_files'][0]['content'], (r.ROOT/'bootstrap.sh').read_text())
        self.assertEqual(json.loads(data['write_files'][1]['content'])['jit'], "'; $(touch /tmp/unsafe)\n")

    def test_multiple_gpus_are_rejected(self):
        with self.assertRaises(r.CIError):
            r.instance_spec('p', 's', 'gpu-h200-sxm', '8gpu-128vcpu-1600gb', 'image', REPO, NAME, 'jit', 0)

    def test_unique_attempts_and_repositories(self):
        self.assertNotEqual(NAME, r.run_name(REPO, 123, 2))
        self.assertNotEqual(NAME, r.run_name('elsewhere/solwa', 123, 1))
        with self.assertRaises(r.CIError):
            r.run_name(REPO, '1;rm', 1)

    def test_exact_cleanup_preserves_other_attempts_and_unowned_resources(self):
        other = r.run_name(REPO, 123, 2)
        own = resource('instance')
        cloud = FakeCloud([own, resource('instance', other), resource('instance', repo='other/repo')],
                          [resource('disk')])
        r.cleanup(cloud, FakeGitHub(), name=NAME)
        self.assertEqual(len(cloud.deleted), 2)
        self.assertEqual(len(cloud.data['instance']), 2)
        r.cleanup(cloud, FakeGitHub(), name=NAME)  # idempotent
        self.assertEqual(len(cloud.deleted), 2)

    def test_reaper_only_deletes_expired_owned_resources(self):
        fresh = resource('instance', 'fresh', now=10000)
        malformed = resource('instance', 'malformed')
        malformed['metadata']['labels']['expires-at'] = 'not-an-integer'
        cloud = FakeCloud([resource('instance'), fresh, malformed,
                           resource('instance', repo='other/repo')])
        r.cleanup(cloud, FakeGitHub(), now=10000)
        self.assertEqual(len(cloud.deleted), 1)
        self.assertEqual(len(cloud.data['instance']), 3)

    def test_failed_vm_delete_never_deletes_its_disk_or_runner(self):
        cloud = FakeCloud([resource('instance')], [resource('disk')], fail_delete=True)
        github = FakeGitHub()
        github.entries = [{'name': NAME, 'id': 123}]
        with self.assertRaises(r.CIError):
            r.cleanup(cloud, github, name=NAME)
        self.assertFalse(cloud.deleted)
        self.assertFalse(github.deleted)

    def test_ready_runner_produces_exact_sha_and_unique_label(self):
        cloud, github = FakeCloud(), FakeGitHub(online=True)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'GITHUB_OUTPUT': str(Path(temp)/'out'), 'NEBIUS_SUBNET_ID': 'subnet'
        }), contextlib.redirect_stdout(io.StringIO()):
            r.provision(cloud, github, 'main', 123, 1)
            outputs = (Path(temp)/'out').read_text()
        self.assertIn('test_sha=' + 'a'*40, outputs)
        self.assertIn('runner_label=' + NAME, outputs)
        self.assertFalse(cloud.deleted)

    def test_lost_create_response_is_cleaned_up_by_labels(self):
        cloud, github = FakeCloud(fail_create=True), FakeGitHub()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'GITHUB_OUTPUT': str(Path(temp)/'out'), 'NEBIUS_SUBNET_ID': 'subnet'
        }), self.assertRaises(r.CIError):
            r.provision(cloud, github, 'main', 123, 1)
        self.assertFalse(cloud.data['instance'])
        self.assertEqual(github.deleted, ['actions/runners/123'])

    def test_boot_timeout_cleans_up(self):
        cloud, github = FakeCloud(), FakeGitHub()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'GITHUB_OUTPUT': str(Path(temp)/'out'), 'NEBIUS_SUBNET_ID': 'subnet'
        }), self.assertRaises(r.CIError):
            r.provision(cloud, github, 'main', 123, 1, ready_timeout=0)
        self.assertFalse(cloud.data['instance'])
        self.assertTrue(github.deleted)

    def _provision_with_fake_clock(self, cloud, github, ready_timeout):
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'GITHUB_OUTPUT': str(Path(temp)/'out'), 'NEBIUS_SUBNET_ID': 'subnet'
        }), contextlib.redirect_stdout(out), self.assertRaises(r.CIError) as ctx:
            r.provision(cloud, github, 'main', 123, 1, ready_timeout=ready_timeout,
                        poll=10, clock=lambda: now[0], sleep=sleep)
        return str(ctx.exception), out.getvalue()

    def test_stopped_vm_is_nudged_and_timeout_reports_state_and_logs(self):
        cloud, github = FakeCloud(states=['STARTING', 'STOPPED']), FakeGitHub()
        error, out = self._provision_with_fake_clock(cloud, github, ready_timeout=60)
        self.assertIn(('compute', 'instance', 'start'), cloud.calls)
        self.assertIn('VM state: STOPPED', out)
        self.assertIn('console log line', out)
        self.assertIn('never reached RUNNING', error)
        self.assertFalse(cloud.data['instance'])

    def test_running_vm_without_runner_points_at_bootstrap(self):
        cloud, github = FakeCloud(states=['RUNNING']), FakeGitHub()
        error, out = self._provision_with_fake_clock(cloud, github, ready_timeout=600)
        self.assertNotIn(('compute', 'instance', 'start'), cloud.calls)
        self.assertIn('bootstrap problem', error)

    def test_slow_start_is_left_alone(self):
        cloud, github = FakeCloud(states=['CREATING', 'STARTING']), FakeGitHub()
        error, out = self._provision_with_fake_clock(cloud, github, ready_timeout=r.READY_TIMEOUT)
        self.assertNotIn(('compute', 'instance', 'start'), cloud.calls)
        self.assertIn('never reached RUNNING', error)

    def test_failed_vm_is_restarted_until_timeout_and_error_is_logged(self):
        cloud, github = FakeCloud(states=['STARTING', 'STOPPED']), FakeGitHub()
        cloud.ops = [{'id': 'op1', 'description': 'Start Instance', 'finished_at': 't',
                      'status': {'code': 8, 'details': [{'code': 'NotEnoughResources'}]}}]
        error, out = self._provision_with_fake_clock(cloud, github, ready_timeout=600)
        starts = cloud.calls.count(('compute', 'instance', 'start'))
        self.assertGreater(starts, 3)  # no fixed limit
        self.assertLessEqual(starts, 600 // r.NUDGE_GAP)
        self.assertEqual(out.count('failed: NotEnoughResources'), 1)
        self.assertIn('last Nebius error NotEnoughResources', error)

    def test_no_start_while_an_operation_is_running(self):
        # Right after create the VM reads STOPPED while the create still runs.
        cloud, github = FakeCloud(states=['STOPPED']), FakeGitHub()
        cloud.ops = [{'id': 'create', 'description': 'Create Instance'}]
        self._provision_with_fake_clock(cloud, github, ready_timeout=600)
        self.assertNotIn(('compute', 'instance', 'start'), cloud.calls)

    def test_no_start_while_reconciling(self):
        cloud, github = FakeCloud(states=['STOPPED']), FakeGitHub()
        cloud.reconciling = True
        self._provision_with_fake_clock(cloud, github, ready_timeout=600)
        self.assertNotIn(('compute', 'instance', 'start'), cloud.calls)

    def test_existing_vm_prevents_new_allocation(self):
        cloud = FakeCloud([resource('instance')])
        with self.assertRaises(r.CIError):
            r.provision(cloud, FakeGitHub(), 'main', 123, 1)
        self.assertIsNone(cloud.payload)

    def test_orphan_runner_reaped_without_vm(self):
        github = FakeGitHub()
        github.entries = [{'name': NAME, 'id': 123, 'busy': False}]
        r.cleanup(FakeCloud(), github, now=10000)
        self.assertEqual(github.deleted, ['actions/runners/123'])

    def test_preflight_aggregates_missing_configuration(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(r.CIError) as ctx:
            r.preflight()
        self.assertIn('NEBIUS_CREDENTIALS_JSON', str(ctx.exception))
        self.assertIn('NEBIUS_SUBNET_ID', str(ctx.exception))

    def test_cli_errors_do_not_leak_payload_or_stderr(self):
        cloud = r.Nebius.__new__(r.Nebius)
        with tempfile.TemporaryDirectory() as temp:
            cloud.directory = Path(temp)
            cloud.config = Path(temp)/'config'
            result = subprocess.CompletedProcess([], 1, '', 'PRIVATE-KEY JIT-SECRET')
            with patch.object(subprocess, 'run', return_value=result), self.assertRaises(r.CIError) as ctx:
                cloud.call('compute', 'instance', 'create', payload={'secret': 'JIT-SECRET'})
            self.assertNotIn('SECRET', str(ctx.exception))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_profile_create_tolerates_plain_text_output(self):
        result = subprocess.CompletedProcess([], 0, 'profile "solwa-ci" configured and activated\n', '')
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {'NEBIUS_CREDENTIALS_JSON': '{"subject-credentials": {}}'}), \
                patch.object(subprocess, 'run', return_value=result) as run:
            cloud = r.Nebius(temp, 'project')
            self.assertIn('profile', run.call_args.args[0])
            self.assertEqual(cloud.project, 'project')

    def test_cloud_list_paginates(self):
        cloud = r.Nebius.__new__(r.Nebius)
        cloud.project = 'project'
        with patch.object(cloud, 'call', side_effect=[{'items': [1], 'next_page_token': 'next'}, {'items': [2]}]) as call:
            self.assertEqual(list(cloud.resources('instance')), [1, 2])
            self.assertIn('next', call.call_args.args)


if __name__ == '__main__':
    unittest.main()
