import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import FacultyDashboard from './FacultyDashboard';

jest.mock('../services/apiService', () => ({
    getLlmBaseUrl: () => '/llm',
}));

describe('FacultyDashboard', () => {
    beforeEach(() => {
        global.fetch = jest.fn();
        URL.createObjectURL = jest.fn(() => 'blob:major-graph');
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    it('requests a major graph and renders the returned image', async () => {
        fetch.mockResolvedValue({
            ok: true,
            blob: jest.fn().mockResolvedValue(new Blob(['png'], { type: 'image/png' })),
        });

        render(<FacultyDashboard />);

        fireEvent.change(screen.getByLabelText(/Enter Major Name/i), {
            target: { value: 'Computer Science, B.S. (Abington)' },
        });
        fireEvent.click(screen.getByRole('button', { name: /Generate Graph/i }));

        await waitFor(() => {
            expect(fetch).toHaveBeenCalledWith('/llm/generate-major-graph', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ major_name: 'Computer Science, B.S. (Abington)' }),
            });
        });
        expect(await screen.findByText(/Graph generated successfully/i)).toBeInTheDocument();
        expect(screen.getByAltText(/Major Requirements Graph/i)).toHaveAttribute('src', 'blob:major-graph');
    });
});
