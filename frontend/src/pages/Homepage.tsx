import Header from '../components/Header'
import '../styles/Homepage.css'

const retrievalCards = [
    {
        href: '/kis',
        label: 'KIS',
        title: 'Find specific moments based on visual descriptions.',
        prompt: '"A red car drifting around a tight corner at night."',
        icon: 'kis',
    },
    {
        href: '/qna',
        label: 'QNA',
        title: 'Get semantic answers to natural language questions.',
        prompt: '"How many times did the suspect check their phone?"',
        icon: 'qna',
    },
    {
        href: '/trake',
        label: 'TRAKE',
        title: 'Locate sequences of multiple temporal events.',
        prompt: '"Person enters room THEN opens safe THEN leaves hurriedly."',
        icon: 'trake',
    },
]

function Homepage() {
    return (
        <>
            <Header />
            <main className="homepage">
                <section className="homepage__intro" aria-labelledby="homepage-title">
                    <span className="system-label">SYSTEM_READY</span>
                    <h1 id="homepage-title">Start a new retrieval session_</h1>
                </section>

                <section className="retrieval-grid" aria-label="Retrieval modes">
                    {retrievalCards.map((card) => (
                        <a className="retrieval-card" href={card.href} key={card.label}>
                            <div className="retrieval-card__topline">
                                <span className="card-label">{card.label}</span>
                                <span className={`card-icon card-icon--${card.icon}`} aria-hidden="true" />
                            </div>
                            <p className="retrieval-card__title">{card.title}</p>
                            <span className="prompt-label">EXAMPLE_PROMPT</span>
                            <span className="retrieval-card__prompt">{card.prompt}</span>
                        </a>
                    ))}
                </section>
            </main>
        </>
    )
}

export default Homepage