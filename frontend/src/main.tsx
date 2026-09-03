import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import Homepage from './pages/Homepage'
import Kis from './pages/KIS'
import Qna from './pages/QNA'
import Trake from './pages/TRAKE'

function App() {
    switch (window.location.pathname) {
        case '/kis':
            return <Kis />
        case '/qna':
            return <Qna />
        case '/trake':
            return <Trake />
        default:
            return <Homepage />
    }
}

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <App />
    </StrictMode>,
)