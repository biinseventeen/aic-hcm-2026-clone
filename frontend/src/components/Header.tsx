import "./Header.css";

const retrievalModes = [
  { label: "RETRIEVAL-KIS", href: "/kis" },
  { label: "RETRIEVAL-QNA", href: "/qna" },
  { label: "RETRIEVAL-TRAKE", href: "/trake" },
];

function Header() {
  const currentPath = typeof window !== "undefined" ? window.location.pathname : "/kis";

  return (
    <header className="site-header">
      <a className="site-header__logo" href="/" aria-label="AIC home">
        AIC_
      </a>

      <nav className="site-header__nav" aria-label="Retrieval modes">
        {retrievalModes.map((mode) => {
          const isActive = currentPath === mode.href || (mode.href === "/kis" && currentPath === "/");
          return (
            <a
              className={`site-header__menu ${isActive ? "site-header__menu--active" : ""}`}
              href={mode.href}
              key={mode.href}
            >
              {mode.label}
              <span className="site-header__chevron" aria-hidden="true" />
            </a>
          );
        })}
      </nav>

      <a className="site-header__new-search" href="/">
        <span>+</span> NEW SEARCH
      </a>
    </header>
  );
}

export default Header;

