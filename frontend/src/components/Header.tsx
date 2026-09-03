import './Header.css'

const retrievalModes = [
    { label: 'RETRIEVAL-KIS', href: '/kis' },
    { label: 'RETRIEVAL-QNA', href: '/qna' },
    { label: 'RETRIEVAL-TRAKE', href: '/trake' },
]

function Header() {
    return (
        <header className="site-header">
            <a className="site-header__logo" href="/" aria-label="AIC home">
                AIC_
            </a>

            <nav className="site-header__nav" aria-label="Retrieval modes">
                {retrievalModes.map((mode) => (
                    <a className="site-header__menu" href={mode.href} key={mode.href}>
                        {mode.label}
                        <span className="site-header__chevron" aria-hidden="true" />
                    </a>
                ))}
            </nav>
        </header>
    )
}

export default Header
