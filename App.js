// import logo from './logo.svg';
// import './App.css';
// import Login from './Components/Login/Login';

// function App() {
//   return (
//     <div>
//       <Login/>
//     </div>
//   );
// }

// export default App;

import React, { useState } from 'react';
import './App.css';
import Login from './Components/Login/Login';
import Dashboard from './Components/Dashboard/Dashboard';

function App() {
  const [currentUser, setCurrentUser] = useState(null);

  return (
    <div>
      {!currentUser ? (
        <Login onLogin={(username) => setCurrentUser(username)} />
      ) : (
        <Dashboard username={currentUser} onLogout={() => setCurrentUser(null)} />
      )}
    </div>
  );
}

export default App;
