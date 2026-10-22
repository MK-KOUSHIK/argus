import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import App from '../src/App';

describe('Argus Application Navigation & Smoke Test', () => {
  it('renders Argus brand header with tagline and Phase 3 badge', () => {
    render(<App />);
    expect(screen.getAllByText('Argus').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Phase 3')).toBeInTheDocument();
    expect(
      screen.getAllByText(/Understand\. Optimize\. Execute\./i).length
    ).toBeGreaterThanOrEqual(1);
  });

  it('renders status indicators for local LLM and main LLM gateway', () => {
    render(<App />);
    expect(screen.getAllByText(/Local LLM:/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Main LLM:/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Disconnected \(Phase 4\)/i)).toBeInTheDocument();
  });

  it('switches between Workspace and System Status tabs', () => {
    render(<App />);
    // Defaults to Workspace tab
    expect(
      screen.getByText('Optimize Input Tokens & Preserve Constraints')
    ).toBeInTheDocument();

    // Click System Status tab
    const systemTabBtn = screen.getByRole('button', { name: /system status/i });
    fireEvent.click(systemTabBtn);

    // Verifies Phase 1 system overview is displayed with health check button
    expect(
      screen.getByText('Cost-Efficient LLM Inference Architecture')
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /verify backend health/i })
    ).toBeInTheDocument();

    // Switch back to Workspace tab
    const workspaceTabBtn = screen.getByRole('button', { name: /workspace/i });
    fireEvent.click(workspaceTabBtn);
    expect(
      screen.getByText('Optimize Input Tokens & Preserve Constraints')
    ).toBeInTheDocument();
  });
});
