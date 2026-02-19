import React, { useState, useEffect } from 'react';
import './Dashboard.css';

const Dashboard = ({ username, onLogout }) => {
    const [gestures, setGestures] = useState({});
    const [statusMsg, setStatusMsg] = useState("");

    useEffect(() => {
        fetch(`http://localhost:8000/user/${username}/gestures`)
            .then(res => res.json())
            .then(data => setGestures(data))
            .catch(err => console.error(err));
    }, [username]);

    const handleRecord = async (gestureName, actionName) => {
        setStatusMsg(`Get ready! 3-second countdown starting for: ${actionName}...`);
        
        try {
            await fetch('http://localhost:8000/command/record', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    user_id: username, 
                    gesture_name: gestureName
                })
            });
            
            // Clear message after recording finishes (3s countdown + ~2s recording)
            setTimeout(() => {
                setStatusMsg(`Successfully updated ${actionName}! Model retrained.`);
                setTimeout(() => setStatusMsg(""), 3000);
            }, 6000);

        } catch (error) {
            setStatusMsg("Error connecting to backend.");
        }
    };

    return (
        <div className="dashboard-wrapper">
            <div className="dashboard-header">
                <h2>Welcome, {username}</h2>
                <button className="logout-btn" onClick={onLogout}>Logout</button>
            </div>
            
            {statusMsg && <div className="status-banner">{statusMsg}</div>}

            <div className="dashboard-content">
                <div className="gesture-list">
                    <h3>Your Desktop Controls</h3>
                    <p className="instruction-text">
                        Click "Re-Record" to map a new physical hand movement to the action.
                    </p>
                    
                    {Object.keys(gestures).map((key) => (
                        <div className="gesture-card" key={key}>
                            <div className="gesture-info">
                                <strong>{gestures[key].action}</strong>
                                <span>System ID: {gestures[key].name}</span>
                            </div>
                            <div className="gesture-actions">
                                <button 
                                    className="update-btn" 
                                    onClick={() => handleRecord(key, gestures[key].action)}
                                >
                                    Re-Record
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default Dashboard;