# Dashboard integration and remaining work

Recorded on 2026-10-01. This is a discussion record and unfinished-work list.
Implementation resumed after the user authorized finishing the application and
committing each verified milestone.

## Requirements stated by the user

- The dashboard belongs in the same repository as oMLX.
- Ship the dashboard and inference backend as one bundled application.
- The dashboard server owns the public listener. Dashboard pages and inference
  clients use the same public host and port.
- The inference server is internal. Public inference traffic reaches it through
  the application proxy, rather than a second exposed inference port.
- Keep TanStack Start, shadcn/ui preset b43fOHkIM, Oxfmt, type-aware Oxlint, and
  the native TypeScript compiler.
- Transfer the old admin's useful management functionality with better HCI.
  Chat remains outside the current scope.

## Current implementation and architecture gap

The backend and frontend now share the oMLX repository, with the frontend
in `dashboard/`. The frontend was moved from the incorrect sibling directory.
The old reference remains at
`/tmp/omlx-old-admin-0b07cdd`.

The frontend currently builds a Node server using TanStack Start and Nitro.
Its management gateway handles dashboard sessions and JSON management requests.
It assumes an independently started oMLX server selected by OMLX_API_URL.
It does not yet provide the required public inference proxy or a bundled launcher.
The Python CLI currently binds the inference and management FastAPI application
directly to the configured host and port.

Moving source files alone will not satisfy the bundled application requirement.
Public listener ownership, internal transport, process supervision, authentication,
packaging, configuration, and client launch URLs need to be integrated.

## Proposed runtime for discussion

One `omlx` command will manage the public Node dashboard server and a private
Python inference process. The user selected a command distribution. The gateway
uses native Nitro proxying in production and Vite proxying in development.

- Serve dashboard pages and assets on the public listener.
- Keep dashboard management requests behind the existing administrator session.
- Proxy supported inference protocol paths on the same listener, preserving the
  inference client's credentials and protocol behavior.
- Stream responses without JSON conversion or whole-response buffering. Preserve
  multipart uploads, binary responses, upstream status, and cancellation.
- Keep the internal endpoint on loopback and selected by the launcher. Preserve
  public host and port settings separately from the private transport.
- Make one launcher own readiness, shutdown, and restart of both processes.
- Bundle the production dashboard build and its runtime with the Python backend,
  according to the distribution format selected by the user.
- Public host and port settings must describe the public listener. Integration
  commands must use that public address. The internal transport is implementation
  configuration.

Authentication must account for the public listener even though upstream traffic
arrives locally. Existing loopback authentication bypass behavior must not silently
turn public proxied requests into trusted local requests. The proxy must not grant
ordinary inference clients the dashboard's main-key management privileges.

Backend restart keeps the dashboard running. The launcher, proxy, and distribution
build are being implemented; relocating source does not complete them.

## Unfinished work to preserve

- [x] Integrate the frontend into the oMLX repository; fix scripts, generated
  schema source paths, test paths, ignores, and documentation.
- [ ] Implement the agreed bundled launcher, private backend transport, public
  inference proxy, and distribution build. This is additional work uncovered by
  the clarified architecture, beyond relocating the frontend.
- [ ] Review public authentication and public host/port semantics through the
  private proxy, including all client integration URLs and restart behavior.
- [x] Finish the interrupted `implement_workspace_backend` model move changes:
  whole cached repositories, preserved model identity/settings/profiles, explicit
  drain, file reservations, rediscovery validation, and rollback. Independent
  review verified 65 workspace/API tests and cancellation rollback probes.
- [ ] Finish the final non-chat functionality audit against the old dashboard.
  Other major areas have implementations, but full completion is not yet certified.
- [ ] Inspect the captured desktop/mobile screenshots and complete the HCI review.
- [ ] Re-review the final resource detection changes and completed move changes.
- [ ] Regenerate contracts and rerun the appropriate backend tests, full browser
  suite, format, type-aware lint, native typecheck, and production build after
  integration.
- [ ] Update documentation and the coverage ledger to match the verified result.

## Last verification checkpoint

The previous targeted backend run passed 1,137 tests before the final resource
changes and interrupted move edits. Resource changes subsequently passed focused
checks. The last full browser run passed 49 of 50 tests. Its failing gateway
assertion was corrected and passed alone; the complete suite was not rerun.
Frontend formatting, type-aware lint, native typecheck, and production build passed
at that checkpoint. These results do not certify the interrupted move edits or
the bundled architecture, which has not been implemented.

No real model downloads, weight loading, provider publication, remote deployment,
or OS memory-limit changes were performed as dashboard validation. No commits or
pushes were made. The existing work is preserved.
