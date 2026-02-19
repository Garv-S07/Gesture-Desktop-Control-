// import React,{useState} from 'react'
// import './Login.css'

// import user_icon from '../Assets/person.png'
// import password_icon from '../Assets/password.png'

// export const Login = () => {

//     const[action,setAction] = useState("Sign Up");

//   return (
//     <div className='container'>
//         <div className="header">
//             <div className="text">{action}</div>
//             <div className="underline"></div>
//         </div>
//         <div className="inputs">
//             <div className="input">
//                 <img src={user_icon} alt="user_icon" />
//                 <input type='text' placeholder='Username'/>
//             </div>
//             <div className="input">
//                 <img src={password_icon} alt="password_icon" />
//                 <input type='password' placeholder='Password'/>
//             </div>
//         </div>
//         <div className="submit-container">
//             <div className={action==="Login"?"submit gray":"submit"} onClick={()=>{setAction("Sign Up")}}>Sign Up</div>
//             <div className={action==="Sign Up"?"submit gray":"submit"} onClick={()=>{setAction("Login")}}>Login</div>
//         </div>
//     </div>
//   )
// }

// export default Login

import React, { useState } from 'react'
import './Login.css'

import user_icon from '../Assets/person.png'
import password_icon from '../Assets/password.png'

export const Login = ({ onLogin }) => {
    const [action, setAction] = useState("Login");
    const [username, setUsername] = useState("");

    const handleSubmit = async () => {
        if (!username.trim()) return alert("Please enter a username");

        try {
            const response = await fetch(`http://localhost:8000/login/${username}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                onLogin(username);
            } else {
                alert("Failed to connect profile.");
            }
        } catch (error) {
            alert("Ensure your FastAPI server is running.");
        }
    };

    return (
        <div className='container'>
            <div className="header">
                <div className="text">{action}</div>
                <div className="underline"></div>
            </div>
            <div className="inputs">
                <div className="input">
                    <img src={user_icon} alt="user_icon" />
                    <input 
                        type='text' 
                        placeholder='Username' 
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                    />
                </div>
                <div className="input">
                    <img src={password_icon} alt="password_icon" />
                    <input type='password' placeholder='Password'/>
                </div>
            </div>
            <div className="submit-container">
                <div className={action==="Login"?"submit gray":"submit"} onClick={()=>{setAction("Sign Up")}}>Sign Up</div>
                <div className={action==="Sign Up"?"submit gray":"submit"} onClick={handleSubmit}>Login</div>
            </div>
        </div>
    )
}

export default Login;
