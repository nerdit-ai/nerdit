import { describe, expect, it } from "vitest";
import type { AppBindings } from "../api/queries";
import type { Database, Model, Service } from "../api/types";
import { deriveProject, deriveProjects, kindHomePath, projectsBoundTo } from "./projects";

// Two model rows (mirror fixtures.ts SAMPLE_MODELS): one ready, one still
// pulling. Reused so readiness comes from the shipped lib/bindings helpers.
const MODELS: Model[] = [
  {
    id: "mdl1cccc0003",
    name: "ollama-llama3-1-8b",
    model: "llama3.1:8b",
    backend: "ollama",
    status: "running",
    desired_state: "running",
    model_pulled: true,
    gpu_count: 1,
    gpu_ids: ["GPU-A100-001"],
    gpu_utilization: { "GPU-A100-001": 42 },
    endpoint: "http://127.0.0.1:38100/v1",
    created_at: "2026-07-04T08:00:00+00:00"
  },
  {
    id: "mdl2dddd0004",
    name: "ollama-qwen2-5-3b",
    model: "qwen2.5:3b",
    backend: "ollama",
    status: "running",
    desired_state: "running",
    model_pulled: false,
    gpu_count: 0,
    gpu_ids: [],
    gpu_utilization: {},
    endpoint: null,
    created_at: "2026-07-04T08:30:00+00:00"
  }
];

const DATABASES: Database[] = [
  {
    id: "db1aaaa0001",
    name: "pg",
    backend: "postgres",
    status: "running",
    desired_state: "running",
    db_ready: true,
    endpoint: "127.0.0.1:38200",
    created_at: "2026-07-14T08:00:00+00:00"
  }
];

function makeService(overrides: Partial<Service> = {}): Service {
  return {
    id: "svc1aaaa0001",
    name: "my-app",
    status: "running",
    desired_state: "running",
    kind: "service",
    image: "nerdit-app/my-app:3",
    gpu_count: 0,
    gpu_ids: [],
    restart_policy: "always",
    restart_count: 0,
    health_check: null,
    container_id: "ctr-my-app",
    created_at: "2026-07-04T10:00:00+00:00",
    started_at: "2026-07-04T10:01:00+00:00",
    finished_at: null,
    exit_code: null,
    error_class: null,
    error_message: null,
    submitted_via: "cli",
    endpoint: null,
    rollback_available: false,
    build_version: 3,
    last_deploy: { version: 3, action: "redeploy", phase: "healthy" },
    ...overrides
  };
}

function bindings(overrides: Partial<AppBindings> = {}): AppBindings {
  return { name: "my-app", ai: null, db: null, error: null, ...overrides };
}

describe("deriveProject — status precedence (failed > deploying > stopped > degraded > attention > running)", () => {
  it("running: app running, no bindings", () => {
    const p = deriveProject(makeService(), bindings(), MODELS, DATABASES);
    expect(p.status).toBe("running");
    expect(p.statusDetail).toBe("App running.");
  });

  it("failed: the app's own status is failed (beats a healthy last_deploy)", () => {
    const p = deriveProject(
      makeService({ status: "failed", error_message: "boom" }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("failed");
    expect(p.statusDetail).toBe("boom");
  });

  it("failed: a failed deploy with nothing serving (status stopped)", () => {
    const p = deriveProject(
      makeService({
        status: "stopped",
        last_deploy: { phase: "failed", reason: "container exited (1)" }
      }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("failed");
    expect(p.statusDetail).toBe("Last deploy failed: container exited (1)");
  });

  it("deploying: a mid-flight phase beats running (redeploy over an old image)", () => {
    const p = deriveProject(
      makeService({ status: "running", last_deploy: { phase: "building" } }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("deploying");
    expect(p.statusDetail).toBe("Building the new image.");
  });

  it("deploying: queued and launching phases too", () => {
    expect(
      deriveProject(makeService({ last_deploy: { phase: "queued" } }), bindings(), MODELS, DATABASES)
        .status
    ).toBe("deploying");
    expect(
      deriveProject(
        makeService({ last_deploy: { phase: "launching" } }),
        bindings(),
        MODELS,
        DATABASES
      ).status
    ).toBe("deploying");
  });

  it("stopped: status stopped", () => {
    const p = deriveProject(
      makeService({ status: "stopped", last_deploy: { phase: "healthy" } }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("stopped");
    expect(p.statusDetail).toBe("App stopped.");
  });

  it("stopped: desired_state stopped even while the row still reads running", () => {
    const p = deriveProject(
      makeService({ status: "running", desired_state: "stopped" }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("stopped");
  });

  it("degraded: the app's own health is degraded", () => {
    const p = deriveProject(makeService({ status: "degraded" }), bindings(), MODELS, DATABASES);
    expect(p.status).toBe("degraded");
    expect(p.statusDetail).toBe("App degraded, health checks failing.");
  });

  it("deploying: a building row with no deploy phase (direct POST /services)", () => {
    const p = deriveProject(
      makeService({ status: "building", last_deploy: null }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("deploying");
    expect(p.statusDetail).toBe("App starting.");
  });

  it("stopped: terminal completed and cancelled exits", () => {
    expect(
      deriveProject(
        makeService({ status: "completed", last_deploy: null }),
        bindings(),
        MODELS,
        DATABASES
      )
    ).toMatchObject({ status: "stopped", statusDetail: "App exited." });
    expect(
      deriveProject(
        makeService({ status: "cancelled", last_deploy: null }),
        bindings(),
        MODELS,
        DATABASES
      )
    ).toMatchObject({ status: "stopped", statusDetail: "App cancelled." });
  });

  it("an unhandled status never claims running (honest attention fallback)", () => {
    const p = deriveProject(
      makeService({ status: "paused", last_deploy: null }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("attention");
    expect(p.statusDetail).toBe("App paused.");
  });

  it("restarting: a transient reconcile transition reads as attention", () => {
    const p = deriveProject(makeService({ status: "restarting" }), bindings(), MODELS, DATABASES);
    expect(p.status).toBe("attention");
    expect(p.statusDetail).toBe("App restarting.");
  });

  it("restarting + a mid-flight phase still deploys (phase checks come first)", () => {
    const p = deriveProject(
      makeService({ status: "restarting", last_deploy: { phase: "building" } }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("deploying");
  });
});

describe("deriveProject — the redeploy-failure nuance (D6)", () => {
  it("running app + last_deploy.phase failed → attention, not failed (old image serving)", () => {
    const p = deriveProject(
      makeService({ status: "running", last_deploy: { phase: "failed", reason: "build error" } }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("attention");
    expect(p.statusDetail).toBe("Last deploy failed. Previous version still serving.");
  });

  it("degraded app + failed deploy stays degraded (degraded outranks attention)", () => {
    const p = deriveProject(
      makeService({ status: "degraded", last_deploy: { phase: "failed" } }),
      bindings(),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("degraded");
  });
});

describe("deriveProject — binding readiness → attention (D6, naming tooltip)", () => {
  it("a still-pulling bound model puts a running app in attention, naming the model", () => {
    const p = deriveProject(
      makeService(),
      bindings({ ai: { default: { provider: "ollama", model: "qwen2.5:3b" } } }),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("attention");
    expect(p.statusDetail).toBe("Waiting on model qwen2.5:3b.");
    expect(p.resources).toHaveLength(1);
    expect(p.resources[0]).toMatchObject({ type: "ai", target: "qwen2.5:3b" });
  });

  it("a ready bound model leaves the project running", () => {
    const p = deriveProject(
      makeService(),
      bindings({ ai: { default: { provider: "ollama", model: "llama3.1:8b" } } }),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("running");
    expect(p.resources[0].readiness.state).toBe("ready");
  });

  it("a missing managed database puts the app in attention, naming the database", () => {
    const p = deriveProject(
      makeService(),
      bindings({ db: { default: { provider: "managed", database: "gone" } } }),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("attention");
    expect(p.statusDetail).toBe("Waiting on database gone.");
  });

  it("a ready managed database leaves the project running", () => {
    const p = deriveProject(
      makeService(),
      bindings({ db: { default: { provider: "managed", database: "pg" } } }),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("running");
  });

  it("an external api binding contributes an unknown (never waiting) resource", () => {
    const p = deriveProject(
      makeService(),
      bindings({
        ai: {
          default: {
            provider: "api",
            base_url: "https://api.openai.com/v1",
            api_key: "${secrets.OPENAI_KEY}"
          }
        }
      }),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("running");
    expect(p.resources[0].readiness.state).toBe("unknown");
    expect(p.resources[0].target).toBe("https://api.openai.com/v1");
  });

  it("a binding with no target set names the binding rather than an invented target", () => {
    const p = deriveProject(
      makeService(),
      bindings({ ai: { cheap: { provider: "ollama" } } }),
      MODELS,
      DATABASES
    );
    expect(p.status).toBe("attention");
    expect(p.statusDetail).toBe("Waiting on the cheap model binding.");
  });
});

describe("deriveProject — loading / error bindings contribute nothing", () => {
  it("undefined bindings (still loading): status from the service alone, no resources", () => {
    const p = deriveProject(makeService(), undefined, MODELS, DATABASES);
    expect(p.status).toBe("running");
    expect(p.resources).toEqual([]);
    expect(p.configError).toBeNull();
  });

  it("a fan-out error sets configError and contributes no resources", () => {
    const err = new Error("config fetch failed");
    const p = deriveProject(
      makeService(),
      bindings({ error: err, ai: { default: { provider: "ollama", model: "qwen2.5:3b" } } }),
      MODELS,
      DATABASES
    );
    // Despite an ai entry, the error path drops resources — the app still renders.
    expect(p.status).toBe("running");
    expect(p.resources).toEqual([]);
    expect(p.configError).toBe(err);
  });
});

describe("deriveProjects", () => {
  it("maps each service to its matching bindings by name, preserving order", () => {
    const services = [makeService({ name: "a" }), makeService({ name: "b", status: "degraded" })];
    const apps = [
      bindings({ name: "b", ai: { default: { provider: "ollama", model: "llama3.1:8b" } } }),
      bindings({ name: "a", ai: { default: { provider: "ollama", model: "qwen2.5:3b" } } })
    ];
    const projects = deriveProjects(services, apps, MODELS, DATABASES);
    expect(projects.map((p) => p.name)).toEqual(["a", "b"]);
    expect(projects[0].status).toBe("attention"); // a → qwen still pulling
    expect(projects[1].status).toBe("degraded"); // b → degraded outranks its ready model
  });

  it("returns an empty array for no services", () => {
    expect(deriveProjects([], [], MODELS, DATABASES)).toEqual([]);
  });
});

describe("projectsBoundTo — reverse index (D9)", () => {
  // Two apps bound to the same model ref; order follows the projects array.
  const aiProjects = deriveProjects(
    [makeService({ name: "chat" }), makeService({ name: "api" })],
    [
      bindings({ name: "chat", ai: { default: { provider: "ollama", model: "llama3.1:8b" } } }),
      bindings({ name: "api", ai: { default: { provider: "ollama", model: "llama3.1:8b" } } })
    ],
    MODELS,
    DATABASES
  );

  it("ai: names every project bound to a model ref, preserving order", () => {
    expect(projectsBoundTo(aiProjects, "ai", "llama3.1:8b")).toEqual(["chat", "api"]);
  });

  it("db: names projects bound to a managed database by name", () => {
    const projects = deriveProjects(
      [makeService({ name: "web" })],
      [bindings({ name: "web", db: { default: { provider: "managed", database: "pg" } } })],
      MODELS,
      DATABASES
    );
    expect(projectsBoundTo(projects, "db", "pg")).toEqual(["web"]);
  });

  it("no match → empty array", () => {
    expect(projectsBoundTo(aiProjects, "ai", "qwen2.5:3b")).toEqual([]);
  });

  it("dedupes a project that binds the same target twice", () => {
    const projects = deriveProjects(
      [makeService({ name: "twin" })],
      [
        bindings({
          name: "twin",
          ai: {
            default: { provider: "ollama", model: "llama3.1:8b" },
            cheap: { provider: "ollama", model: "llama3.1:8b" }
          }
        })
      ],
      MODELS,
      DATABASES
    );
    expect(projectsBoundTo(projects, "ai", "llama3.1:8b")).toEqual(["twin"]);
  });

  it("discriminates type: an ai target never matches a db query", () => {
    // An app whose ai binding targets "pg" (a model ref that happens to collide
    // with a database name) must not surface for a db query on "pg".
    const projects = deriveProjects(
      [makeService({ name: "x" })],
      [bindings({ name: "x", ai: { default: { provider: "ollama", model: "pg" } } })],
      MODELS,
      DATABASES
    );
    expect(projectsBoundTo(projects, "db", "pg")).toEqual([]);
    expect(projectsBoundTo(projects, "ai", "pg")).toEqual(["x"]);
  });
});

describe("kindHomePath", () => {
  it("falls back to the label path when the row carries no project field", () => {
    expect(kindHomePath({ kind: "service", name: "my-app" })).toBe("/projects/my-app");
    expect(kindHomePath({ kind: "service", name: "my-app", project: null, service: null })).toBe(
      "/projects/my-app"
    );
  });

  it("URL-encodes the service name", () => {
    expect(kindHomePath({ kind: "service", name: "a b/c" })).toBe("/projects/a%20b%2Fc");
  });

  it("routes the service whose label is the project name to the project route", () => {
    expect(
      kindHomePath({ kind: "service", name: "asso", project: "asso", service: "web" })
    ).toBe("/projects/asso");
  });

  it("routes any other service of a project by its FIELDS, never the label", () => {
    // The label deliberately disagrees with the fields: a parser would say `api`/`asso`.
    expect(
      kindHomePath({ kind: "service", name: "api--asso", project: "other", service: "worker" })
    ).toBe("/projects/other/services/worker");
  });

  it("routes models and databases to their inventory pages", () => {
    expect(kindHomePath({ kind: "model", name: "ollama-llama3-1-8b", project: null })).toBe(
      "/models"
    );
    expect(kindHomePath({ kind: "database", name: "appdb" })).toBe("/databases");
  });

  it("falls back to the project grid for an unknown kind (batch has no page)", () => {
    expect(kindHomePath({ kind: "mystery" as Service["kind"], name: "x" })).toBe("/projects");
    expect(kindHomePath({ kind: "batch", name: "some-job" })).toBe("/projects");
  });
});
