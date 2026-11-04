import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import UnifiedChat from './UnifiedChat';
import { clearConversation, generateResponse, setupEnvironment } from '../services/apiService';

jest.mock('react-router-dom', () => ({
    useNavigate: () => jest.fn(),
}));

jest.mock('../services/apiService', () => ({
    clearConversation: jest.fn(),
    generateResponse: jest.fn(),
    setupEnvironment: jest.fn(),
}));

describe('UnifiedChat', () => {
    beforeAll(() => {
        Element.prototype.scrollIntoView = jest.fn();
    });

    beforeEach(() => {
        sessionStorage.clear();
        jest.clearAllMocks();
        clearConversation.mockResolvedValue({ ok: true });
        generateResponse.mockResolvedValue({
            response: 'The prerequisites for CMPSC 488 are CMPSC 487W.',
            conversationHistory: [{ role: 'assistant', content: 'CMPSC 487W' }],
        });
        setupEnvironment.mockResolvedValue('Ready');
    });

    it('sends a question with session history and renders the advisor response', async () => {
        render(<UnifiedChat />);

        expect(screen.getByText('Leo')).toBeInTheDocument();

        fireEvent.change(screen.getByPlaceholderText(/Ask anything/i), {
            target: { value: 'what are the prerequisites of cmpsc 488' },
        });
        fireEvent.click(screen.getByRole('button', { name: /send/i }));

        await waitFor(() => {
            expect(generateResponse).toHaveBeenCalledWith(
                'what are the prerequisites of cmpsc 488',
                [],
                expect.any(String)
            );
        });
        expect(await screen.findByText(/CMPSC 487W/i)).toBeInTheDocument();
    });

    it('clears local and server conversation state', async () => {
        window.confirm = jest.fn(() => true);
        render(<UnifiedChat />);

        fireEvent.change(screen.getByPlaceholderText(/Ask anything/i), {
            target: { value: 'plan my next semester' },
        });
        fireEvent.click(screen.getByRole('button', { name: /send/i }));

        expect(await screen.findByText(/CMPSC 487W/i)).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: /clear/i }));

        await waitFor(() => expect(clearConversation).toHaveBeenCalledWith(expect.any(String)));
        await waitFor(() => expect(screen.getByText(/Plan smarter/i)).toBeInTheDocument());
        expect(screen.queryByText(/CMPSC 487W/i)).not.toBeInTheDocument();
    });

    it('shows an upload error when transcript setup fails', async () => {
        jest.spyOn(console, 'error').mockImplementation(() => {});
        setupEnvironment.mockRejectedValueOnce(new Error('upload failed'));
        const { container } = render(<UnifiedChat />);

        const fileInput = container.querySelector('input[type="file"]');
        const file = new File(['pdf'], 'transcript.pdf', { type: 'application/pdf' });
        fireEvent.change(fileInput, { target: { files: [file] } });

        expect(await screen.findByText(/Failed to upload transcript/i)).toBeInTheDocument();
    });
});
