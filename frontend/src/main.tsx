import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { CapabilityProvider } from "./lib/capabilities";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <CapabilityProvider>
        <App />
      </CapabilityProvider>
    </BrowserRouter>
  </StrictMode>,
);
