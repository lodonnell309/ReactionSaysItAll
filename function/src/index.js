import React from 'react';
import ReactDOM from 'react-dom/client';
import IPhoneMessageApp from './components/IPhoneMessageApp';

// Create a root and render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <IPhoneMessageApp />
  </React.StrictMode>
);