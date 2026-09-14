"""Crash-safe checkpoint persistence and integrity verification."""

import hashlib
import json
import os
import re
import stat
import tempfile


CHECKSUM_SUFFIX = '.sha256'
METADATA_SUFFIX = '.metadata.json'


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path):
    with open(path, 'rb') as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temp(temp_path, destination, immutable):
    if immutable:
        # A hard link publishes atomically and refuses an existing destination.
        os.link(temp_path, destination)
        os.unlink(temp_path)
    else:
        if os.path.islink(destination):
            raise RuntimeError('refusing to replace checkpoint symlink: %s' % destination)
        os.replace(temp_path, destination)


def atomic_write_text(path, text, immutable=False):
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(prefix='.%s.' % os.path.basename(path), suffix='.tmp', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp(temp_path, path, immutable)
        _fsync_directory(directory)
    except Exception:
        if os.path.lexists(temp_path):
            os.unlink(temp_path)
        raise


def atomic_save_checkpoint(
        checkpoint, filename, immutable=False, write_checksum=False, expected_step=None):
    """Write a torch checkpoint without exposing a partial destination file."""
    import torch

    filename = os.path.abspath(filename)
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    if os.path.islink(filename):
        raise RuntimeError('refusing to write checkpoint through symlink: %s' % filename)
    if immutable and os.path.lexists(filename):
        raise FileExistsError('refusing to overwrite immutable checkpoint: %s' % filename)
    if immutable and write_checksum and (os.path.lexists(filename + CHECKSUM_SUFFIX) or os.path.lexists(filename + METADATA_SUFFIX)):
        raise FileExistsError('refusing to write checkpoint with existing integrity sidecar: %s' % filename)

    descriptor, temp_path = tempfile.mkstemp(
        prefix='.%s.' % os.path.basename(filename), suffix='.tmp', dir=directory)
    os.close(descriptor)
    try:
        torch.save(checkpoint, temp_path)
        if os.path.getsize(temp_path) <= 0:
            raise RuntimeError('torch produced an empty checkpoint: %s' % temp_path)
        _fsync_file(temp_path)
        digest = sha256_file(temp_path) if write_checksum else None
        _publish_temp(temp_path, filename, immutable)
        _fsync_directory(directory)
        if digest is not None:
            if expected_step is None:
                match = re.search(r'_checkpoint_(\d+)\.(?:pt|pth)$', os.path.basename(filename))
                expected_step = int(match.group(1)) if match else None
            atomic_write_text(
                filename + CHECKSUM_SUFFIX,
                '%s  %s\n' % (digest, os.path.basename(filename)),
                immutable=immutable,
            )
            metadata = {
                'filename': os.path.basename(filename),
                'sha256': digest,
                'size_bytes': os.path.getsize(filename),
                'step': expected_step,
            }
            atomic_write_text(
                filename + METADATA_SUFFIX,
                json.dumps(metadata, indent=2, sort_keys=True) + '\n',
                immutable=immutable,
            )
        return digest
    except Exception:
        if os.path.lexists(temp_path):
            os.unlink(temp_path)
        raise


def validate_checkpoint_file(
        path, require_checksum=False, require_metadata=False, expected_step=None):
    """Reject links, empty files, and files that disagree with their checksum."""
    if require_metadata:
        require_checksum = True
    path = os.path.abspath(path)
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise ValueError('checkpoint is missing: %s (%s)' % (path, exc))
    if not stat.S_ISREG(mode):
        raise ValueError('checkpoint must be a regular non-symlink file: %s' % path)
    if os.path.getsize(path) <= 0:
        raise ValueError('checkpoint is empty: %s' % path)

    checksum_path = path + CHECKSUM_SUFFIX
    if not os.path.isfile(checksum_path) or os.path.islink(checksum_path):
        if require_checksum:
            raise ValueError('checkpoint checksum is missing: %s' % checksum_path)
        return None
    with open(checksum_path, encoding='ascii') as handle:
        fields = handle.read().strip().split()
    if not fields:
        raise ValueError('invalid checkpoint checksum file: %s' % checksum_path)
    expected = fields[0].lower()
    if len(expected) != 64 or any(char not in '0123456789abcdef' for char in expected):
        raise ValueError('invalid checkpoint checksum file: %s' % checksum_path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError('checkpoint checksum mismatch: %s' % path)
    metadata_path = path + METADATA_SUFFIX
    if not os.path.isfile(metadata_path) or os.path.islink(metadata_path):
        if require_metadata:
            raise ValueError('checkpoint metadata is missing: %s' % metadata_path)
        return actual
    try:
        with open(metadata_path, encoding='utf-8') as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError('invalid checkpoint metadata file: %s (%s)' % (metadata_path, exc))
    if metadata.get('filename') != os.path.basename(path):
        raise ValueError('checkpoint metadata filename mismatch: %s' % path)
    if metadata.get('sha256') != actual:
        raise ValueError('checkpoint metadata hash mismatch: %s' % path)
    if metadata.get('size_bytes') != os.path.getsize(path):
        raise ValueError('checkpoint metadata size mismatch: %s' % path)
    if expected_step is not None and metadata.get('step') != expected_step:
        raise ValueError('checkpoint metadata step mismatch: %s' % path)
    return actual


def atomic_copy_checkpoint(
        source, destination, require_checksum=False, require_metadata=False,
        expected_step=None):
    """Create a regular-file alias without ever publishing a partial copy."""
    digest = validate_checkpoint_file(
        source, require_checksum=require_checksum,
        require_metadata=require_metadata, expected_step=expected_step)
    source = os.path.abspath(source)
    destination = os.path.abspath(destination)
    if source == destination or os.path.realpath(source) == os.path.realpath(destination):
        raise ValueError('checkpoint source and destination are identical: %s' % source)
    if os.path.lexists(destination):
        try:
            if os.path.samefile(source, destination):
                raise ValueError(
                    'checkpoint source and destination share the same inode: %s' % source)
        except OSError:
            pass
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix='.%s.' % os.path.basename(destination), suffix='.tmp', dir=directory)
    try:
        with open(source, 'rb') as input_handle, os.fdopen(descriptor, 'wb') as output_handle:
            while True:
                chunk = input_handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temp_path, destination)
    except Exception:
        if os.path.lexists(temp_path):
            os.unlink(temp_path)
        raise
    _fsync_directory(directory)
    if digest is None:
        digest = sha256_file(source)
    atomic_write_text(
        destination + CHECKSUM_SUFFIX,
        '%s  %s\n' % (digest, os.path.basename(destination)),
    )
    metadata = {
        'filename': os.path.basename(destination),
        'sha256': digest,
        'size_bytes': os.path.getsize(destination),
        'step': expected_step,
    }
    atomic_write_text(
        destination + METADATA_SUFFIX,
        json.dumps(metadata, indent=2, sort_keys=True) + '\n',
    )
    validate_checkpoint_file(
        destination, require_checksum=True, require_metadata=True,
        expected_step=expected_step)
    return digest
