"""Compatibility shim package for local development.

Some modules import using the package name `ovos_voice_agent` while the
directory in-tree is named `ovos-voice-agent`. This small shim adjusts the
package search path so Python can import the in-repo source without
installing the package.
"""
import os
__all__ = []
# Prepend the in-repo folder to the package __path__ so imports work.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
alt = os.path.join(repo_root, 'ovos-voice-agent')
if os.path.isdir(alt):
    __path__.insert(0, alt)