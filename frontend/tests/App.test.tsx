import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import App from '../src/App';
import { apiService } from '../src/services/api';

describe('Argus Application Navigation & Smoke Test', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(apiService, 'getOllamaStatus').mockResolvedValue({
      connected: true,
      base_url: 'http://127.0.0.1:11434',
      model: 'qwen2.5:1.5b',
      model_installed: true,
      available_models: ['qwen2.5:1.5b'],
      error: null,
    });
    vi.spyOn(apiService, 'getHealth').mockResolvedValue({
      status: 'ok',
      app_name: 'Argus',
      version: '0.1.0',
      timestamp: new Date().toISOString(),
      environment: 'development',
    });
  });

  it('renders Argus brand header with tagline and Optimizer badge', async () => {
    render(<App />);
    expect(await screen.findByText('Optimizer')).toBeInTheDocument();
    expect(screen.getAllByText('Argus').length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText(/Understand\. Optimize\. Execute\./i).length
    ).toBeGreaterThanOrEqual(1);
  });

  it('renders status indicators for local LLM and backend API', async () => {
    render(<App />);
    expect((await screen.findAllByText(/Local LLM:/i)).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/API/i).length).toBeGreaterThanOrEqual(1);
  });

  it('switches between Workspace and System Status tabs', async () => {
    render(<App />);
    // Defaults to Workspace tab
    expect(
      await screen.findByText('Optimize Input Tokens & Preserve Constraints')
    ).toBeInTheDocument();

    // Click System Status tab
    const systemTabBtn = screen.getByRole('button', { name: /system status/i });
    fireEvent.click(systemTabBtn);

    // Verifies system status overview is displayed with health check button
    expect(
      await screen.findByText('System Status & Provider Gateway')
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /verify backend health/i })
    ).toBeInTheDocument();

    // Switch back to Workspace tab
    const workspaceTabBtn = screen.getByRole('button', { name: /workspace/i });
    fireEvent.click(workspaceTabBtn);
    expect(
      await screen.findByText('Optimize Input Tokens & Preserve Constraints')
    ).toBeInTheDocument();
  });
});
