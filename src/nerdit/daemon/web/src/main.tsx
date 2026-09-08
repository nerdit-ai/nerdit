import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Navigate, createBrowserRouter, RouterProvider } from "react-router-dom";
import "./styles/globals.css";
import { AppShell } from "./shell/AppShell";
import { ToastViewport } from "./components/ToastViewport";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Settings } from "./pages/Settings";
import { Login } from "./pages/Login";
import Projects from "./pages/Projects";
import ProjectDetail from "./pages/ProjectDetail";
import ServiceRedirect from "./pages/ServiceRedirect";
import Models from "./pages/Models";
import Databases from "./pages/Databases";
import Audit from "./pages/Audit";
import Tokens from "./pages/Tokens";
import { getStoredToken } from "./lib/auth";
import { initTheme } from "./lib/theme";

const queryClient = new QueryClient();

function RequireAuth({ children }: { children: React.ReactNode }) {
  if (!getStoredToken()) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}

// The route map of design guidelines §2. Two rules govern it:
//
// - The apps list is the home page and renders AT "/" — not a redirect to
//   "/projects". A dashboard whose landing URL bounces is one more thing to
//   explain, and "/" is the address an operator types.
// - Every retired URL keeps working. `/projects`, `/store`, `/services` and
//   `/hardware` were all linkable and bookmarkable, so each becomes a replace
//   redirect rather than a 404. `/audit` → `/activity` is the same rename seen
//   from the other side. `/services/:ident` stays the kind-aware shim. The
//   catch-all LAST child is the general form of that rule: any other retired
//   address (`/bindings/ai`, `/bindings/db`, and whatever is retired next)
//   lands on the apps list instead of the router's raw error screen.
//
// The Hardware and Store PAGES are gone (folded into Settings, and into the
// New app flow, respectively) and their files are deleted.
const router = createBrowserRouter([
  { path: "/login", element: <Login /> },
  {
    path: "/",
    element: (
      <RequireAuth>
        <AppShell />
      </RequireAuth>
    ),
    children: [
      { index: true, element: <Projects /> },
      { path: "projects", element: <Navigate to="/" replace /> },
      { path: "projects/:name/:tab?", element: <ProjectDetail /> },
      { path: "services", element: <Navigate to="/" replace /> },
      { path: "services/:ident", element: <ServiceRedirect /> },
      { path: "store", element: <Navigate to="/" replace /> },
      { path: "hardware", element: <Navigate to="/settings" replace /> },
      { path: "models", element: <Models /> },
      { path: "databases", element: <Databases /> },
      { path: "tokens", element: <Tokens /> },
      { path: "activity", element: <Audit /> },
      { path: "audit", element: <Navigate to="/activity" replace /> },
      { path: "settings", element: <Settings /> },
      // Must stay LAST: a catch-all matches anything the rows above did not.
      { path: "*", element: <Navigate to="/" replace /> }
    ]
  }
]);

// Before render: the first paint is already the operator's theme, with no
// light-then-dark flash on a dark-preferring machine.
initTheme();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <RouterProvider router={router} />
      </ErrorBoundary>
      <ToastViewport />
    </QueryClientProvider>
  </React.StrictMode>
);
