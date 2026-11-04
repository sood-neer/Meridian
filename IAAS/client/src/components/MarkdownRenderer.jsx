import React from 'react';
import '../styles/MarkdownRenderer.css';

// Allow course codes from 1-3 digits with optional trailing letter (e.g. SOC 1, MATH 22, CMPSC 221, MATH 140A)
const COURSE_CODE_RE = /\b([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\b/g;
const SEMESTER_HEADER_RE = /^\s*Semester\s+\d+\b[^\n]*$/i;
const TOTAL_CREDITS_RE = /^\s*Total\s+Credits?\s*:\s*([\d.]+)\s*$/i;
const PREREQ_PAREN_RE = /\(prerequisites?:\s*([^)]+)\)/i;
const COMPACT_SEMESTER_LINE_RE = /^\s*["']?\s*(Semester\s+\d+)\s*:\s*(.*?)\s*Total\s+Credits?\s*:\s*([\d.]+)\s*["']?\s*$/i;
const COMPACT_SEMESTER_ITEM_START_RE =
    /(?=\b(?:[A-Z]{2,8}(?:\/[A-Z]{2,8})?\s+\d{1,3}[A-Z]?\s*(?::|\(prerequisites?:)|(?:GQ|GN|GH|GS|GA|GWS|GHW|US|IL)\s+course\b|Open\s+slot\b|Free\s+Elective\b|Option\s+Elective\b))/gi;

// Strip annotations the user has explicitly asked us never to render: when the
// system couldn't retrieve a prerequisite, just drop the parenthetical entirely
// (and the leading space that precedes it) so the course code stands alone.
// Must be applied BEFORE block parsing so it can't sneak through any path.
const STRIP_UNKNOWN_PREREQ_RE =
    /\s*\((?:enforced\s+)?prerequisites?:\s*(?:not\s+retrieved|not\s+listed[^)]*|unknown|unavailable|n\/?a)\s*\)/gi;
const stripUnknownPrereqAnnotations = (text) =>
    typeof text === 'string' ? text.replace(STRIP_UNKNOWN_PREREQ_RE, '') : text;

const splitCompactSemesterItems = (body) => {
    if (!body) return [];
    const starts = [];
    let m;
    const startRegex = new RegExp(COMPACT_SEMESTER_ITEM_START_RE.source, 'gi');
    while ((m = startRegex.exec(body)) !== null) {
        starts.push(m.index);
        if (startRegex.lastIndex === m.index) startRegex.lastIndex += 1;
    }
    if (!starts.length) return [body.trim()].filter(Boolean);
    if (starts[0] !== 0 && body.slice(0, starts[0]).replace(/[ ;,-]/g, '').trim()) {
        starts.unshift(0);
    }
    return starts
        .map((start, idx) => body.slice(start, starts[idx + 1] || body.length).trim().replace(/[ ;]+$/g, ''))
        .filter(Boolean);
};

const canonicalizeSemesterBlocks = (text) => {
    if (typeof text !== 'string' || !text) return text;
    return text.split('\n').map((rawLine) => {
        const match = COMPACT_SEMESTER_LINE_RE.exec(rawLine.trim());
        if (!match) return rawLine;
        const [, semester, body, credits] = match;
        const lines = [`${semester}:`];
        splitCompactSemesterItems(body).forEach((item) => lines.push(`- ${item}`));
        lines.push(`Total Credits: ${credits}`);
        return lines.join('\n');
    }).join('\n');
};

/* -------------------------------------------------------------------------- */
/*  Inline formatting                                                         */
/* -------------------------------------------------------------------------- */

// `emphasizeCodes` is the single flag for "this is a real recommendation":
//   - true  → bold course codes AND show "(prerequisites: ...)" annotations
//   - false → plain course codes AND strip prereq annotations entirely
// Used so prereqs only show inside semester-card course lines, never in the
// intro, prose, headers, lists, or notes.
const STRIP_ALL_PREREQ_RE = /\s*\((?:enforced\s+)?prerequisites?:\s*[^)]+\)/gi;

const renderInlineSegments = (text, baseKey, emphasizeCodes = false) => {
    if (!text) return null;
    if (typeof text !== 'string') return text;

    if (!emphasizeCodes) {
        // Outside a recommendation block: drop every prereq parenthetical
        // (and its leading space) before rendering plain text.
        return renderCourseCodeSegments(text.replace(STRIP_ALL_PREREQ_RE, ''), `${baseKey}-tail`, false);
    }

    const out = [];
    let cursor = 0;
    let m;
    const prereqRegex = new RegExp(PREREQ_PAREN_RE.source, 'gi');
    while ((m = prereqRegex.exec(text)) !== null) {
        if (m.index > cursor) {
            out.push(...renderCourseCodeSegments(text.slice(cursor, m.index), `${baseKey}-pre-${m.index}`, true));
        }
        const value = m[1].trim();
        out.push(
            <span key={`${baseKey}-prereq-${m.index}`} className={`mdx-prereq ${classifyPrereq(value)}`}>
                · prereq: <span className="mdx-prereq-value">{value}</span>
            </span>
        );
        cursor = prereqRegex.lastIndex;
    }
    if (cursor < text.length) {
        out.push(...renderCourseCodeSegments(text.slice(cursor), `${baseKey}-tail`, true));
    }
    return out;
};

const renderCourseCodeSegments = (text, baseKey, emphasizeCodes) => {
    if (!text) return [];
    if (!emphasizeCodes) return [text]; // plain text, no <span>
    const parts = [];
    let cursor = 0;
    let m;
    COURSE_CODE_RE.lastIndex = 0;
    while ((m = COURSE_CODE_RE.exec(text)) !== null) {
        if (m.index > cursor) {
            parts.push(text.slice(cursor, m.index));
        }
        parts.push(
            <span key={`${baseKey}-code-${m.index}`} className="mdx-code">
                {m[1]}&nbsp;{m[2]}
            </span>
        );
        cursor = COURSE_CODE_RE.lastIndex;
    }
    if (cursor < text.length) parts.push(text.slice(cursor));
    return parts;
};

const classifyPrereq = (value) => {
    const v = (value || '').toLowerCase().trim();
    if (!v) return 'is-unknown';
    if (v === 'none' || v === 'no prerequisites' || v.startsWith('none')) return 'is-none';
    if (v.includes('not retrieved') || v.includes('not listed')) return 'is-unknown';
    return 'is-required';
};

/* -------------------------------------------------------------------------- */
/*  Block-level structure                                                     */
/* -------------------------------------------------------------------------- */

const stripBoldMarkers = (s) => s.replace(/\*\*/g, '');

const isBoldOnlyHeader = (line) => /^\*\*[^*]+\*\*\s*:?\s*$/.test(line.trim());
const isBulletLine = (line) => /^[-*•]\s+/.test(line.trim());
const isNumberedLine = (line) => /^\d+\.\s+/.test(line.trim());

const splitBlocks = (text) => {
    const lines = (text || '').split('\n');
    const blocks = [];
    let i = 0;

    while (i < lines.length) {
        const raw = lines[i];
        const trimmed = raw.trim();

        if (!trimmed) { i++; continue; }

        // Semester block: "Semester N: ..." through "Total Credits: N"
        if (SEMESTER_HEADER_RE.test(trimmed)) {
            const block = { kind: 'semester', header: trimmed, items: [], total: null, extras: [] };
            i++;
            while (i < lines.length) {
                const ln = lines[i];
                const t = ln.trim();
                if (SEMESTER_HEADER_RE.test(t)) break;
                const totalMatch = TOTAL_CREDITS_RE.exec(t);
                if (totalMatch) {
                    block.total = totalMatch[1];
                    i++;
                    break;
                }
                if (!t) { i++; continue; }
                if (isBulletLine(t) || isNumberedLine(t)) {
                    block.items.push(t.replace(/^[-*•]\s+/, '').replace(/^\d+\.\s+/, ''));
                } else {
                    block.extras.push(t);
                }
                i++;
            }
            blocks.push(block);
            continue;
        }

        if (isBoldOnlyHeader(trimmed)) {
            blocks.push({ kind: 'header', text: stripBoldMarkers(trimmed).replace(/:$/, '') });
            i++;
            continue;
        }

        if (isBulletLine(trimmed) || isNumberedLine(trimmed)) {
            const items = [];
            while (i < lines.length) {
                const t = (lines[i] || '').trim();
                if (!t) break;
                if (!(isBulletLine(t) || isNumberedLine(t))) break;
                items.push(t.replace(/^[-*•]\s+/, '').replace(/^\d+\.\s+/, ''));
                i++;
            }
            blocks.push({ kind: 'list', items });
            continue;
        }

        // Paragraph: collect contiguous non-empty, non-special lines
        const paraLines = [];
        while (i < lines.length) {
            const t = (lines[i] || '').trim();
            if (!t) break;
            if (SEMESTER_HEADER_RE.test(t) || isBoldOnlyHeader(t) || isBulletLine(t) || isNumberedLine(t)) break;
            paraLines.push(t);
            i++;
        }
        if (paraLines.length) blocks.push({ kind: 'paragraph', text: paraLines.join(' ') });
    }

    return blocks;
};

/* -------------------------------------------------------------------------- */
/*  Block renderers                                                           */
/* -------------------------------------------------------------------------- */

const renderHeader = (text, key) => (
    <h3 key={key} className="mdx-header">
        {renderInlineSegments(stripBoldMarkers(text), key)}
    </h3>
);

const renderParagraph = (text, key) => (
    <p key={key} className="mdx-paragraph">
        {renderInlineSegments(stripBoldMarkers(text), key)}
    </p>
);

const renderList = (items, key) => (
    <ul key={key} className="mdx-list">
        {items.map((item, idx) => (
            <li key={`${key}-li-${idx}`} className="mdx-list-item">
                <span className="mdx-list-bullet" aria-hidden="true" />
                <span className="mdx-list-text">
                    {renderInlineSegments(stripBoldMarkers(item), `${key}-li-${idx}`)}
                </span>
            </li>
        ))}
    </ul>
);

const renderSemester = (block, key) => (
    <div key={key} className="mdx-semester">
        <div className="mdx-semester-head">
            <div className="mdx-semester-title">{stripBoldMarkers(block.header.replace(/:$/, ''))}</div>
            {block.total && (
                <div className="mdx-semester-credits">
                    <span className="mdx-semester-credits-num">{block.total}</span>
                    <span className="mdx-semester-credits-label">credits</span>
                </div>
            )}
        </div>
        <ul className="mdx-semester-courses">
            {block.items.map((item, idx) => (
                <li key={`${key}-c-${idx}`} className="mdx-semester-course">
                    <span className="mdx-semester-bullet" aria-hidden="true" />
                    <span className="mdx-semester-course-text">
                        {renderInlineSegments(stripBoldMarkers(item), `${key}-c-${idx}`, true)}
                    </span>
                </li>
            ))}
        </ul>
        {block.extras.length > 0 && (
            <div className="mdx-semester-extras">
                {block.extras.map((line, idx) => (
                    <div key={`${key}-x-${idx}`} className="mdx-semester-note">
                        {renderInlineSegments(stripBoldMarkers(line), `${key}-x-${idx}`)}
                    </div>
                ))}
            </div>
        )}
    </div>
);

/* -------------------------------------------------------------------------- */

const MarkdownRenderer = ({ content }) => {
    if (!content) return null;
    // Strip leading "#" headers (kept simple — they become bold).
    // Then drop any "(prerequisites: not retrieved)" / "(prerequisites: not
    // listed in our course data)" annotations so they never reach the user.
    const cleaned = stripUnknownPrereqAnnotations(
        canonicalizeSemesterBlocks(content)
            .replace(/^#{1,6}\s+/gm, '**$&')
            .replace(/\*\*#{1,6}\s+/g, '**')
    );
    const blocks = splitBlocks(cleaned);

    return (
        <div className="mdx">
            {blocks.map((b, idx) => {
                const key = `mdx-${idx}`;
                switch (b.kind) {
                    case 'semester':  return renderSemester(b, key);
                    case 'header':    return renderHeader(b.text, key);
                    case 'list':      return renderList(b.items, key);
                    case 'paragraph':
                    default:          return renderParagraph(b.text, key);
                }
            })}
        </div>
    );
};

export default MarkdownRenderer;
