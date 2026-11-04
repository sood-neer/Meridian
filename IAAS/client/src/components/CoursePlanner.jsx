import React, { useState } from 'react';
import '../styles/Dashboard.css'; // Ensure consistent styling
import { generateResponse, setupEnvironment } from '../services/apiService';

const CoursePlanner = ({ setDisableButton }) => {
    const [graduationPlan, setGraduationPlan] = useState(''); // Centralized state
    const [loading, setLoading] = useState(false);
    const [generateEnabled, setGenerateEnable] = useState(false);
    const [showInitializing, setShowInitializing] = useState(false);

    const getUserId = () => {
        try {
            const user = JSON.parse(localStorage.getItem('user') || '{}');
            return user.id || user.email || 'default';
        } catch {
            return 'default';
        }
    };

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        initializeEnvironment(file);
    };
    
    const initializeEnvironment = async (file) => {
        try {
            setShowInitializing(true)
            const formData = new FormData();
            formData.append('transcript', file); 
    
            const message = await setupEnvironment(formData, getUserId()); 
            console.log('Environment initialized:', message);
            setGenerateEnable(true);
            setShowInitializing(false)
        } catch (error) {
            console.error('Error initializing environment:', error);
            setGraduationPlan('An error occurred while generating the graduation plan.');
        }
    };

    const generateGraduationPlan = async () => {
        setLoading(true);
        try {
            const result = await generateResponse(
                'Generate a Graduation Plan based on my major from what-if report',
                [],
                getUserId()
            );

            setGraduationPlan(result.response || 'No plan generated.');
            setDisableButton(false);
        } catch (error) {
            console.error('Error generating graduation plan:', error);
            setGraduationPlan('An error occurred while generating the graduation plan.');
        } finally {
            setLoading(false);
        }
    };
   
    return (
        <div className="planner">
            <h3>Course Planner</h3>

            {/* File Upload Section */}
            <div className="form-section">
                <label htmlFor="transcript">Upload Your Transcript (PDF):</label>
                <input
                    type="file"
                    id="transcript"
                    accept=".pdf"
                    onChange={handleFileUpload}
                />

                {/* Generate Graduation Plan Button */}
                <button onClick={generateGraduationPlan} disabled={!generateEnabled || loading}>
                    {/* {loading ? 'Generating...' : 'Generate Graduation Plan'} */}
                    {showInitializing ? 'Initializing...' : loading ? 'Generating...': 'Generate Graduation Plan'}
                </button>

                {/* Graduation Plan Output */}
                {graduationPlan && (
                    <textarea
                        readOnly
                        value={graduationPlan}
                        className="graduation-plan-output"
                        placeholder="Graduation plan will appear here..."
                    />
                )}
            </div>
        </div>
    );
};

export default CoursePlanner;
