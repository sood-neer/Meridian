import React from 'react';
import { render, screen } from '@testing-library/react';
import MarkdownRenderer from './MarkdownRenderer';

const getTextWithNormalizedNbsp = (expected) => (_content, element) =>
    element?.textContent?.replace(/\u00a0/g, ' ') === expected;

describe('MarkdownRenderer', () => {
    it('renders real prerequisite annotations inside semester course rows', () => {
        render(
            <MarkdownRenderer
                content={[
                    'Semester 1:',
                    '- CMPSC 488 (prerequisites: CMPSC 487W)',
                    '- ENGL 202D (prerequisites: ENGL 15)',
                    'Total Credits: 6',
                ].join('\n')}
            />
        );

        expect(screen.getByText('CMPSC 487W')).toBeInTheDocument();
        expect(screen.getByText('ENGL 15')).toBeInTheDocument();
        expect(screen.getByText(getTextWithNormalizedNbsp('ENGL 202D'))).toBeInTheDocument();
    });

    it('strips unknown prerequisite annotations from all rendered output', () => {
        render(
            <MarkdownRenderer
                content={[
                    'Semester 1:',
                    '- CMPSC 131 (prerequisites: not retrieved)',
                    'Total Credits: 3',
                    '',
                    'CMPSC 488 (prerequisites: unknown) is still available for planning discussion.',
                ].join('\n')}
            />
        );

        expect(screen.queryByText(/not retrieved/i)).not.toBeInTheDocument();
        expect(screen.queryByText(/unknown/i)).not.toBeInTheDocument();
    });

    it('turns compact quoted semester lines into readable semester cards', () => {
        const { container } = render(
            <MarkdownRenderer
                content={
                    '"Semester 1: PSU 1: First-Year Seminar Abington (prerequisites: None) CMPSC 131: Programming and Computation I (prerequisites: None) MATH 140: Calculus I (prerequisites: None) GQ course — quantification — examples: STAT 200, MATH 110 Total Credits: 16"'
                }
            />
        );

        expect(container.querySelectorAll('.mdx-semester')).toHaveLength(1);
        expect(screen.getByText('Semester 1')).toBeInTheDocument();
        expect(screen.getByText('16')).toBeInTheDocument();
        expect(screen.getByText(getTextWithNormalizedNbsp('PSU 1'))).toBeInTheDocument();
        expect(screen.getByText(getTextWithNormalizedNbsp('CMPSC 131'))).toBeInTheDocument();
        expect(screen.getByText(/GQ course/i)).toBeInTheDocument();
    });
});
