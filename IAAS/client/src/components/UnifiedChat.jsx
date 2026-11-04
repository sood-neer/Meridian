import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { generateResponse, setupEnvironment, clearConversation } from '../services/apiService';
import MarkdownRenderer from './MarkdownRenderer';
import '../styles/UnifiedChat.css';

import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';
import AccountTreeRoundedIcon from '@mui/icons-material/AccountTreeRounded';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import SendRoundedIcon from '@mui/icons-material/SendRounded';
import PersonRoundedIcon from '@mui/icons-material/PersonRounded';
import SchoolRoundedIcon from '@mui/icons-material/SchoolRounded';
import MenuBookRoundedIcon from '@mui/icons-material/MenuBookRounded';
import InsightsRoundedIcon from '@mui/icons-material/InsightsRounded';
import LightbulbRoundedIcon from '@mui/icons-material/LightbulbRounded';
import ErrorOutlineRoundedIcon from '@mui/icons-material/ErrorOutlineRounded';
import ScheduleRoundedIcon from '@mui/icons-material/ScheduleRounded';

const FEATURE_CARDS = [
    {
        icon: <MenuBookRoundedIcon fontSize="small" />,
        title: 'Course lookups',
        blurb: 'Credits, prerequisites, and descriptions from the academic catalog.',
    },
    {
        icon: <SchoolRoundedIcon fontSize="small" />,
        title: 'Major requirements',
        blurb: 'Full degree audits, prescribed sequences, and gen-ed buckets.',
    },
    {
        icon: <InsightsRoundedIcon fontSize="small" />,
        title: 'Graduation planning',
        blurb: 'Personalized 8-semester roadmaps that respect prereqs.',
    },
    {
        icon: <LightbulbRoundedIcon fontSize="small" />,
        title: 'Smart recommendations',
        blurb: 'Next-best courses tuned to your transcript and interests.',
    },
];

const SUGGESTED_PROMPTS = [
    {
        label: '🎓 NIT Hamirpur B.Tech CSE',
        prompt: 'I am a B.Tech CSE student at NIT Hamirpur with a 7.8 CGPA. What courses should I take in my upcoming semester and how do I plan my departmental electives?'
    },
    {
        label: '📚 Backlogs & Supplementary Exams',
        prompt: 'I have a backlog from last semester at NIT Hamirpur. What is the policy for clearing it via supplementary exams or special terms?'
    },
    {
        label: '💼 T&P Placement Eligibility',
        prompt: 'What are the minimum CGPA and academic eligibility criteria for campus placement drives (T&P / CCD) at NIT Hamirpur?'
    },
    {
        label: '📅 Semester Registration & Credits',
        prompt: 'How many minimum and maximum credits can an NIT Hamirpur student register for in a semester, and how does the grading scale work?'
    },
    {
        label: 'Prerequisites for CMPSC 131',
        prompt: 'What are the prerequisites for CMPSC 131?'
    },
    {
        label: 'Course Repeat Policy',
        prompt: 'What is the course repeat and late drop policy?'
    }
];

const UnifiedChat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [transcriptUploaded, setTranscriptUploaded] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [conversationHistory, setConversationHistory] = useState([]);
    const chatEndRef = useRef(null);
    const fileInputRef = useRef(null);
    const textareaRef = useRef(null);
    const navigate = useNavigate();

    const getSessionId = () => {
        let sessionId = sessionStorage.getItem('session_id');
        if (!sessionId) {
            sessionId = typeof crypto !== 'undefined' && crypto.randomUUID
                ? crypto.randomUUID()
                : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
            sessionStorage.setItem('session_id', sessionId);
        }
        return sessionId;
    };

    useEffect(() => {
        const sessionId = getSessionId();
        const savedHistory = sessionStorage.getItem(`chat_history_${sessionId}`);
        const savedMessages = sessionStorage.getItem(`chat_messages_${sessionId}`);
        const savedTranscriptFlag = sessionStorage.getItem(`transcript_uploaded_${sessionId}`);
        if (savedHistory) {
            try { setConversationHistory(JSON.parse(savedHistory)); } catch (_) {}
        }
        if (savedMessages) {
            try {
                const parsedMessages = JSON.parse(savedMessages);
                const messagesWithDates = parsedMessages.map(msg => ({
                    ...msg,
                    timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date()
                }));
                setMessages(messagesWithDates);
            } catch (_) {}
        }
        if (savedTranscriptFlag === 'true') setTranscriptUploaded(true);
    }, []);

    useEffect(() => {
        const sessionId = getSessionId();
        if (conversationHistory.length > 0) {
            sessionStorage.setItem(`chat_history_${sessionId}`, JSON.stringify(conversationHistory));
        }
        if (messages.length > 0) {
            sessionStorage.setItem(`chat_messages_${sessionId}`, JSON.stringify(messages));
        }
        sessionStorage.setItem(`transcript_uploaded_${sessionId}`, transcriptUploaded ? 'true' : 'false');
    }, [conversationHistory, messages, transcriptUploaded]);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    // Auto-grow textarea
    useEffect(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = 'auto';
        el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
    }, [input]);

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsUploading(true);
        setError('');

        try {
            const formData = new FormData();
            formData.append('transcript', file);
            const sessionId = getSessionId();
            const message = await setupEnvironment(formData, sessionId);
            console.log('Environment initialized:', message);
            setTranscriptUploaded(true);

            const uploadMessage = {
                id: Date.now(),
                role: 'assistant',
                content:
                    "Transcript received. Leo can now tailor graduation plans, course picks, and audits to your record. What would you like to dig into first?",
                timestamp: new Date(),
                kind: 'transcript-ack',
            };
            setMessages(prev => [...prev, uploadMessage]);
        } catch (err) {
            console.error('Error uploading transcript:', err);
            setError('Failed to upload transcript. Please try again.');
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const handlePromptClick = (text) => {
        setInput(text);
        if (textareaRef.current) {
            textareaRef.current.focus();
        }
    };

    const handleSendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = {
            id: Date.now(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setError('');
        setIsLoading(true);

        try {
            const result = await generateResponse(
                userMessage.content,
                conversationHistory,
                getSessionId()
            );
            const aiMessage = {
                id: Date.now() + 1,
                role: 'assistant',
                content: result.response,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, aiMessage]);
            setConversationHistory(result.conversationHistory || []);
        } catch (err) {
            console.error('Error sending message:', err);
            setError('Failed to get a response. Please try again.');
            setMessages(prev => [...prev, {
                id: Date.now() + 1,
                role: 'assistant',
                content: 'Something went sideways reaching Leo. Try again in a moment.',
                timestamp: new Date(),
                isError: true
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const handleClearChat = async () => {
        if (!window.confirm('Clear the current conversation? Your transcript upload will also be removed.')) {
            return;
        }
        const sessionId = getSessionId();
        try {
            await clearConversation(sessionId);
        } catch (err) {
            console.error('Error clearing conversation on server:', err);
            // Continue with local cleanup even if the server call failed so the UI
            // never ends up in a half-cleared state.
        }

        setMessages([]);
        setConversationHistory([]);
        setTranscriptUploaded(false);
        setInput('');
        setError('');
        setIsLoading(false);
        setIsUploading(false);

        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }

        sessionStorage.removeItem(`chat_history_${sessionId}`);
        sessionStorage.removeItem(`chat_messages_${sessionId}`);
        sessionStorage.removeItem(`transcript_uploaded_${sessionId}`);
    };

    const formatTimestamp = (ts) => {
        const d = ts instanceof Date ? ts : new Date(ts);
        return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    };

    return (
        <div className="iaas-shell">
            <header className="iaas-header">
                <div className="iaas-brand">
                    <span className="iaas-brand-mark">
                        <AutoAwesomeRoundedIcon fontSize="small" />
                    </span>
                    <div className="iaas-brand-text">
                        <span className="iaas-brand-title">Leo</span>
                        <span className="iaas-brand-sub">NIT Hamirpur · AI Academic Advisor</span>
                    </div>
                </div>

                <div className="iaas-header-actions">
                    <button
                        type="button"
                        className="iaas-chip"
                        onClick={() => navigate('/faculty-dashboard')}
                        title="Open the prerequisite graph"
                    >
                        <AccountTreeRoundedIcon fontSize="small" />
                        <span>Course Graph</span>
                    </button>

                    <label
                        className={`iaas-chip ${transcriptUploaded ? 'iaas-chip-ok' : ''} ${isUploading ? 'iaas-chip-busy' : ''}`}
                        title={transcriptUploaded ? 'Transcript on file' : 'Upload your transcript (PDF)'}
                    >
                        {transcriptUploaded ? (
                            <CheckCircleRoundedIcon fontSize="small" />
                        ) : (
                            <UploadFileRoundedIcon fontSize="small" />
                        )}
                        <span>{transcriptUploaded ? 'Transcript on file' : (isUploading ? 'Uploading…' : 'Upload transcript')}</span>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".pdf"
                            onChange={handleFileUpload}
                            disabled={isUploading}
                        />
                    </label>

                    {messages.length > 0 && (
                        <button
                            type="button"
                            className="iaas-chip iaas-chip-danger"
                            onClick={handleClearChat}
                            title="Clear the conversation"
                        >
                            <DeleteOutlineRoundedIcon fontSize="small" />
                            <span>Clear</span>
                        </button>
                    )}
                </div>
            </header>

            <main className="iaas-stream">
                {messages.length === 0 && (
                    <section className="iaas-hero">
                        <div className="iaas-hero-badge">
                            <AutoAwesomeRoundedIcon fontSize="small" />
                            <span>Your personal academic co-pilot</span>
                        </div>
                        <h1 className="iaas-hero-title">
                            Plan smarter. <span>Graduate faster.</span>
                        </h1>
                        <p className="iaas-hero-sub">
                            Ask anything about courses, majors, prerequisites, or graduation planning.
                            Upload your transcript and Leo will personalize every recommendation.
                        </p>
                        <div className="iaas-feature-grid">
                            {FEATURE_CARDS.map((f) => (
                                <div key={f.title} className="iaas-feature-card">
                                    <span className="iaas-feature-icon">{f.icon}</span>
                                    <div className="iaas-feature-copy">
                                        <div className="iaas-feature-title">{f.title}</div>
                                        <div className="iaas-feature-blurb">{f.blurb}</div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="iaas-suggestions">
                            <span className="iaas-suggestions-title">Quick Starter Prompts</span>
                            <div className="iaas-suggestion-chips">
                                {SUGGESTED_PROMPTS.map((item) => (
                                    <button
                                        key={item.label}
                                        type="button"
                                        className="iaas-suggestion-chip"
                                        onClick={() => handlePromptClick(item.prompt)}
                                    >
                                        {item.label}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </section>
                )}

                <div className="iaas-messages">
                    {messages.map((message) => {
                        const isUser = message.role === 'user';
                        const isError = !!message.isError;
                        return (
                            <div
                                key={message.id}
                                className={`iaas-row ${isUser ? 'iaas-row-user' : 'iaas-row-ai'} ${isError ? 'iaas-row-error' : ''}`}
                            >
                                <div className={`iaas-avatar ${isUser ? 'iaas-avatar-user' : 'iaas-avatar-ai'}`}>
                                    {isUser
                                        ? <PersonRoundedIcon fontSize="small" />
                                        : (isError
                                            ? <ErrorOutlineRoundedIcon fontSize="small" />
                                            : <AutoAwesomeRoundedIcon fontSize="small" />)}
                                </div>
                                <div className="iaas-bubble-wrap">
                                    <div className="iaas-bubble-meta">
                                        <span className="iaas-bubble-author">
                                            {isUser ? 'You' : 'Leo'}
                                        </span>
                                        <span className="iaas-bubble-time">
                                            <ScheduleRoundedIcon fontSize="inherit" />
                                            {formatTimestamp(message.timestamp)}
                                        </span>
                                    </div>
                                    <div className={`iaas-bubble ${isUser ? 'iaas-bubble-user' : 'iaas-bubble-ai'} ${isError ? 'iaas-bubble-error' : ''}`}>
                                        {isUser || isError
                                            ? <div className="iaas-plain-text">{message.content}</div>
                                            : <MarkdownRenderer content={message.content} />
                                        }
                                    </div>
                                </div>
                            </div>
                        );
                    })}

                    {isLoading && (
                        <div className="iaas-row iaas-row-ai iaas-row-loading">
                            <div className="iaas-avatar iaas-avatar-ai">
                                <AutoAwesomeRoundedIcon fontSize="small" />
                            </div>
                            <div className="iaas-bubble-wrap">
                                <div className="iaas-bubble iaas-bubble-ai">
                                    <div className="iaas-typing">
                                        <span /><span /><span />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={chatEndRef} />
                </div>
            </main>

            {error && (
                <div className="iaas-banner iaas-banner-error">
                    <ErrorOutlineRoundedIcon fontSize="small" />
                    <span>{error}</span>
                </div>
            )}

            <div className="iaas-composer">
                <div className="iaas-composer-inner">
                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={transcriptUploaded
                            ? "Ask about your plan, next courses, audits, prereqs…"
                            : "Ask anything — or upload your transcript for personalized planning."}
                        disabled={isLoading || isUploading}
                        rows={1}
                        className="iaas-composer-input"
                    />
                    <button
                        type="button"
                        onClick={handleSendMessage}
                        disabled={isLoading || isUploading || !input.trim()}
                        className="iaas-send"
                        aria-label="Send"
                    >
                        <SendRoundedIcon fontSize="small" />
                    </button>
                </div>
                <div className="iaas-composer-hint">
                    <span>Press <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a newline</span>
                </div>
            </div>
        </div>
    );
};

export default UnifiedChat;
