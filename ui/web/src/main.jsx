// main.jsx
import React from "react";
import "./index.css";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import App from "./App.jsx";
import TestBench from "./TestBench.jsx";
import SingularisShowcase from './SingularisShowcase.jsx';

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SingularisShowcase />} />
        <Route path="/dbg" element={<App />} />
        <Route path="/test" element={<TestBench />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
