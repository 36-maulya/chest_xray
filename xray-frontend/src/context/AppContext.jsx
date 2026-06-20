import { createContext, useContext, useState } from "react";

const AppContext = createContext();

export function AppProvider({ children }) {
  const [results,  setResults]  = useState([]);
  const [current,  setCurrent]  = useState(null);

  const addResult = (result) => {
    const entry = {
      ...result,
      id:        Date.now(),
      timestamp: new Date().toLocaleString(),
    };
    setResults((prev) => [entry, ...prev]);
    setCurrent(entry);
    return entry;
  };

  return (
    <AppContext.Provider value={{ results, current, setCurrent, addResult }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  return useContext(AppContext);
}