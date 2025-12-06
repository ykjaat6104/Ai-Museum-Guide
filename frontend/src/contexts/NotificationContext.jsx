import React, { createContext, useContext } from 'react';
import toast from 'react-hot-toast';

const NotificationContext = createContext();

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

export const NotificationProvider = ({ children }) => {

  const showSuccess = (message) => {
    toast.success(message);
  };

  const showError = (message) => {
    toast.error(message);
  };

  const showInfo = (message) => {
    toast(message, {
      icon: '💡',
      style: {
        background: 'var(--primary-600)',
        color: 'var(--neutral-0)',
      },
    });
  };

  const showWarning = (message) => {
    toast(message, {
      icon: '⚠️',
      style: {
        background: 'var(--warning-500)',
        color: 'var(--neutral-0)',
      },
    });
  };

  const showLoading = (message) => {
    return toast.loading(message);
  };

  const dismissLoading = (toastId) => {
    toast.dismiss(toastId);
  };

  const value = {
    showSuccess,
    showError,
    showInfo,
    showWarning,
    showLoading,
    dismissLoading
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};