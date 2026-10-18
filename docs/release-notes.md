# Release notes

The product version is set in one place: the `VERSION` file at the root of the
repository. Both services read it from there and report it: `GET /health` on
`:8051`, which adds `commit` (the git commit the query API started from), and
both capability maps, `GET /schema` and `GET /api/schema`.

A release changes three things in one commit. It bumps `VERSION`, adds an entry
at the top of `CHANGELOG.md`, and moves the `Noetrix version:` line in each agent
skill to the same version. `tests/test_versioning.py` fails when any of them
disagree, when either API reports a different version, or when the changelog's
versions and dates do not descend.

A running console shows its version on the chip in its header, which also counts
the parts of the install that are behind. The chip opens a panel listing those
parts beside these notes. `GET /api/releases` on `:8052` returns the same as JSON:
the version on disk, the checkout's commit, every part of the install (this
console, the query API, the installed agent skills, a packaged bundle) as
`current`, `behind`, `missing`, `not running` or `invalid` with what to do about
it, and the release notes parsed from `CHANGELOG.md`.

The changelog follows, newest first. Its entries use the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) sections, and the
versions follow [Semantic Versioning](https://semver.org/).

<!-- The include below starts at CHANGELOG.md's note on its reconstructed early
     entries. The changelog's title and the two paragraphs before that note say
     what the text above says. If those opening paragraphs change length, move
     the start line so the include still begins on that note. -->

--8<-- "CHANGELOG.md:13"
