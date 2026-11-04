export const getLlmBaseUrl = () => {
    const envUrl = process.env.REACT_APP_LLM_URL;
    if (envUrl) {
        return envUrl.replace(/\/+$/, "");
    }
    if (typeof window !== "undefined") {
        const host = window.location.hostname;
        if (host === "localhost" || host === "127.0.0.1") {
            return "http://127.0.0.1:5002/llm";
        }
    }
    return "/llm";
};

export const getAuthBaseUrl = () => {
    const envUrl = process.env.REACT_APP_AUTH_URL;
    if (envUrl) {
        return envUrl.replace(/\/+$/, "");
    }
    if (typeof window !== "undefined") {
        const host = window.location.hostname;
        if (host === "localhost" || host === "127.0.0.1") {
            return "http://localhost:5001/api";
        }
    }
    return "/api";
};

const requestJson = async (url, options = {}) => {
    const response = await fetch(url, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...(options.headers || {}),
        },
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`);
    }
    return data;
};

const authHeader = () => {
    if (typeof window === "undefined") {
        return {};
    }
    const token = window.localStorage.getItem("token");
    return token ? { Authorization: `Bearer ${token}` } : {};
};

export const loginUser = async ({ email, password }) => {
    return requestJson(`${getAuthBaseUrl()}/login`, {
        method: "POST",
        body: JSON.stringify({ email, password }),
    });
};

export const registerUser = async ({ name, email, password, role, sex, dateOfBirth }) => {
    return requestJson(`${getAuthBaseUrl()}/register`, {
        method: "POST",
        headers: authHeader(),
        body: JSON.stringify({ name, email, password, role, sex, dateOfBirth }),
    });
};

export const generateResponse = async (userQuestion, conversationHistory = [], userId = 'default') => {
    try {
        const runtimeDebug =
            typeof window !== 'undefined'
                ? String(window.localStorage.getItem('debug_trace') || '').toLowerCase() === 'true'
                : false;
        const debugTraceEnabled =
            String(process.env.REACT_APP_DEBUG_TRACE || '').toLowerCase() === 'true' || runtimeDebug;
        const response = await fetch(`${getLlmBaseUrl()}/generate-response`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                user_question: userQuestion,
                conversation_history: conversationHistory,
                user_id: userId,
                debug_trace: debugTraceEnabled
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Response from LLM:", data);
        if (debugTraceEnabled && data.debug_trace) {
            console.log("Execution trace:", data.debug_trace);
        }
        return {
            response: data.graduation_plan || data.response || '',
            conversationHistory: data.conversation_history || conversationHistory
        };
    } catch (error) {
        console.error('Error querying LLM:', error);
        throw error;
    }
};

export const clearConversation = async (userId = 'default') => {
    try {
        const response = await fetch(`${getLlmBaseUrl()}/clear-conversation`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_id: userId }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error clearing conversation:', error);
        throw error;
    }
};

export const setupEnvironment = async (formData, userId = 'default') => {
    try {
        // Append user_id to formData
        formData.append('user_id', userId);
        
        const response = await fetch(`${getLlmBaseUrl()}/setup-environment`, {
            method: 'POST',
            body: formData, // Send the FormData object
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.message; // Return the success message from the backend
    } catch (error) {
        console.error('Error setting up environment:', error);
        throw error;
    }
};
