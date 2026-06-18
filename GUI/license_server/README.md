# License Server Package

This package is the future home of the customer-deployed on-prem floating license server for
`Surrogate Model Training Suite`.

The scaffold is intentionally split into API, CLI, crypto, database, schema, service-bootstrap,
and business-rule layers so the same core logic can be reused by both the local CLI and the HTTP
service.

See:

- `doc/implementation/license_server_repository_scaffold.md`
- `doc/implementation/license_server_implementation_checklist.md`
