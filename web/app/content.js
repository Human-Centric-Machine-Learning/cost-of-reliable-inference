// All explanatory text of the page, kept apart from the application logic.

const CONTENT = {
  primer: [
    {
      title: "The problem",
      html: `A platform routes one question per round to a <b>provider</b>: a language model served under a
      specific configuration. Each provider has an unknown <b>quality</b> and a private <b>cost</b> per query.
      The platform wants the cheapest provider whose quality meets the reliability threshold
      <b>$\\Theta$</b>. This provider is called <b>$i^*$</b>. The platform must learn quality from graded answers
      and cost from providers' standing bids.`,
    },
    {
      title: "The mechanism",
      html: `Each provider is tried once. Later, the platform keeps providers whose <b>optimistic quality
      estimate</b> still reaches $\\Theta$. It selects the lowest <b>score</b>: the standing bid minus a cost
      radius <b>$\\rho(m)$</b>. The radius shrinks with selections, giving less-tested providers an exploration
      bonus. The winner receives a <b>threshold payment</b>, capped at <b>$C_{\\max}$</b>. Under the paper's
      assumptions, bidding the current cost estimate is one-round optimal and undominated.`,
    },
    {
      title: "The baselines",
      html: `The baselines use the same simulated outcomes with different selection rules. Oracle policies
      receive true qualities and costs; the others use observed state. Only policies for which the threshold
      payment is valid are paid. Unpaid policies are compared on their selections.`,
    },
    {
      title: "The ablations",
      html: `Each ablation changes one part of the mechanism: quality filtering, exploration, payment,
      belief slack, cost estimation, or the coupling between answer quality and cost.`,
    },
    {
      title: "How to read the simulation",
      html: `The simulation uses <b>recorded generations</b>: 128 answers per model and question, with
      correctness, token count, and reward score. A <code>model@N</code> provider draws N answers, returns the
      reward model's preferred answer, and pays for all N. Monetary values are in <b>millionths of a
      dollar</b>.<br><br>
      An episode is <b>one shuffled pass over the benchmark</b>. Within a repetition, every policy receives
      the same question order and pre-drawn provider outcomes. Repetitions redraw those outcomes. The
      available horizons are shorter than the paper's identification certificate, so the results describe
      finite-horizon behaviour rather than guaranteed convergence.`,
    },
  ],

  arms: {
    mechanism: {
      label: "Mechanism",
      short: "The paper's rule: quality filter, score = bid − $\\rho(m)$, threshold payment.",
      long: "Keeps the providers whose optimistic quality still reaches $\\Theta$, selects the lowest bid minus exploration radius, pays the threshold price.",
    },
    uniform_random: {
      label: "Uniform random",
      short: "Picks any provider uniformly at random, every round.",
      long: "Uses no quality or cost information, providing a no-learning reference. Unpaid.",
    },
    cheapest_bid: {
      label: "Cheapest bid",
      short: "Always the lowest standing bid, ignoring quality.",
      long: "Uses no quality filter or exploration and can settle on an unqualified provider that appears cheap. Unpaid.",
    },
    quality_greedy: {
      label: "Quality greedy",
      short: "The highest optimistic quality among the eligible, ignoring cost.",
      long: "Uses the same quality filter as the mechanism, then picks the most promising provider regardless of price. Unpaid.",
    },
    oracle_cheapest_qualified: {
      label: "Oracle: cheapest qualified",
      short: "Knows the truth and always selects $i^*$.",
      long: "A cost reference with perfect information. Unpaid.",
    },
    cheapest_price: {
      label: "Cheapest list price",
      short: "The lowest advertised per-token rate, no learning.",
      long: "Selects on the public price list alone, ignoring both quality and realised token counts. Unpaid.",
    },
    oracle_quality_random: {
      label: "Oracle: random qualified",
      short: "Knows the qualified set and picks uniformly inside it.",
      long: "Perfect quality information, no cost information. Unpaid.",
    },
    no_quality_filter: {
      label: "No quality filter",
      short: "The mechanism's score over the whole roster.",
      long: "Removes eligibility, so every provider remains a candidate. The score minimizer still receives the threshold payment.",
    },
    greedy_cheapest_qualified: {
      label: "Greedy cheapest qualified",
      short: "The mechanism without the exploration term.",
      long: "Keeps the quality filter but selects the lowest bid among the eligible instead of the lowest bid minus $\\rho(m)$. It stops exploring once one provider looks cheapest. Unpaid.",
    },
    pay_your_bid: {
      label: "Pay your bid",
      short: "The mechanism's selection, first-price payment.",
      long: "The winner is paid exactly its bid instead of the threshold price. The selection is unchanged; the payment metrics show what the threshold price buys.",
    },
    gamma_zero: {
      label: "$\\gamma = 0$",
      short: "The mechanism with no widening of the cost radius.",
      long: "Uses the unwidened radius whatever the providers' estimation rule. Meaningful together with a shrinkage estimator, where some widening is required.",
    },
    biased_beliefs: {
      label: "Biased beliefs",
      short: "Providers bid their estimate plus a constant offset.",
      long: "Breaks the assumption that bids track the empirical mean cost: every provider adds a fixed fraction of $C_{\\max}$ to its estimate, and no widening bounds the resulting slack.",
    },
    independent_cost_stream: {
      label: "Independent cost stream",
      short: "A query's cost is drawn separately from its answer.",
      long: "In the default stream a provider's cost and correctness on a question come from the same drawn generations. This ablation draws the token subset separately, decoupling the two.",
    },
  },

  params: {
    preset: "The nine shipped environments, each defined by a benchmark, roster, and threshold. Select one as a starting point.",
    benchmark: "Which set of questions the stream is drawn from. GSM8K is grade-school maths, GPQA graduate-level science, AIME competition maths. Qualities differ a lot between them.",
    roster: "The providers competing for the stream. A provider is a base model served best-of-N: model@N draws N recorded answers, keeps the one the reward model prefers, and is charged for all N.",
    theta: "The reliability threshold. A provider is qualified when its true quality is at least $\\Theta$; the mechanism never sees these qualities and must learn who clears the bar. At least two providers must qualify.",
    delta: "The confidence level of the radii. Smaller $\\delta$ means wider radii, so slower elimination and slower price discovery, in exchange for a stronger guarantee that no qualified provider is wrongly ruled out.",
    gamma: "Widening of the cost radius, $\\rho(m) = (1 + \\gamma)\\, C_{\\max}\\, \\beta(m)$. Needed only when providers' bids can drift from their empirical mean cost, as with the shrinkage estimator; 0 is correct for honest empirical-mean bidders.",
    margin: "Providers' list prices are (1 + margin) times cost, so private cost is inferred by deflating the price. When $C_{\\max}$ is pinned, changing the margin rescales costs and bids but not the radius ceiling.",
    c_max: "The most a provider can ever be paid in a round, and the scale of the cost radius. It cannot go below the largest cost any roster provider can realise (the validity floor). A larger $C_{\\max}$ means slower learning for everyone.",
    estimator: "How each provider turns its observed costs into a bid. Empirical mean is the honest baseline; shrinkage pulls towards a prior and needs $\\gamma$ above a computed minimum; biased adds a constant offset and violates the assumption on purpose.",
    arms: "Each arm is one selection rule, or one ablation of the mechanism, run on the same query stream. Hover a name for what it does.",
    repetitions: "Independent redraws of the query stream. Every arm sees the same stream within a repetition. More repetitions give tighter averages but take proportionally longer.",
    t_max: "Rounds per episode. At most the number of questions in the benchmark, since each question is served once; at least the roster size, because every provider is tried once at the start.",
    seed: "The master seed. Together with the environment and the repetition it fixes the question order, the initialisation order, the tie-breaks and the drawn generations. Change it to see a different stream.",
  },

  plots: {
    landscape: "Every candidate provider on this benchmark, cost against quality. Grey dots are candidates not on the roster, the dashed line is the frontier no candidate beats on both axes. Roster providers are coloured: green is $i^*$, blue qualified, orange unqualified. The dotted line is $\\Theta$.",
    radii: "How many selections it takes to resolve each gap. The curve is the quality radius $\\beta(m)$. An unqualified provider is ruled out once $\\beta(m)$ falls below half its quality gap (orange lines); a qualified rival stops winning once the cost radius falls below half its cost gap, drawn here in the same units (blue lines). The grey line marks one pass.",
    overview_scatter: "Each arm as a point: total cost of the episode against realised accuracy, mean over repetitions with one standard deviation as error bars. The dotted line is $\\Theta$. Up and to the left is better.",
    overview_share: "Where each arm's rounds went: the share of selections per provider, averaged over repetitions. Green is $i^*$, blues are other qualified providers, oranges are unqualified ones. An arm that is mostly orange spent the episode on providers below the threshold.",
    istar_share: "The running share of rounds given to $i^*$, the cheapest qualified provider, averaged over repetitions. The oracle sits at one; a learning arm rises as it identifies $i^*$, and an arm that locks onto an unqualified provider stays near zero.",
    provider_share: "For one arm and one repetition: the running share of rounds each provider received. The thick green line is $i^*$; orange lines are unqualified providers. Watch which providers the arm keeps returning to.",
    eligible: "How many providers each arm still considers eligible, round by round, in the first repetition. Arms with a quality filter show steps down as unqualified providers are ruled out; the dotted line is the true number of qualified providers.",
    quality_regret: "Cumulative quality regret: the sum over rounds of how far the selected provider's true quality fell below $\\Theta$ (zero when a qualified provider was selected). Both axes are logarithmic. The dashed and dotted lines are the paper's two bounds for the mechanism; they hold only for the mechanism, the other arms are shown against them for scale.",
    generation_regret: "Cumulative cost regret: the sum over rounds of how much more the selected provider's true expected cost was than $i^*$'s. Same conventions as the quality regret.",
    envelope: "Round by round, what the winner was paid, against the band the theory allows: from the winner's true cost minus its radius up to the second-cheapest qualified cost plus that radius, capped at $C_{\\max}$. The horizontal lines are $i^*$'s cost and the second-cheapest qualified cost.",
    payoff: "The winner's payoff, payment minus realised cost, accumulated over the episode: net in black, split into the rounds where payment covered the realised cost (green) and the rounds where it did not (red). A negative round is a query whose answer ran longer than the price paid for it.",
    payoff_providers: "The same split per provider, sorted by net payoff, with the net as a black dot.",
    q_ucb: "The platform's optimistic quality estimate for each provider in the first repetition. A provider is eligible while its line is above $\\Theta$ (dotted).",
    slack: "How far each provider's bid strayed from its empirical mean cost, relative to the cost radius, at each of its selections. The dotted line is the widening $\\gamma$ the platform allows. Honest empirical-mean bidders sit at zero; the biased estimator grows without bound.",
    sweep: "The same comparison at several thresholds. Each dot is one episode; lines show means over repetitions. Labels along the top give the number of qualified providers.",
    collapse: "The share of episodes in which an arm gave more than half its rounds to one unqualified provider. The exploration term is designed to reduce this risk.",
  },

  glossary: [
    ["$\\Theta$", "The reliability threshold. Qualified means true quality at least $\\Theta$."],
    ["$i^*$", "The cheapest qualified provider: what the platform ideally selects every round."],
    ["Quality q", "A provider's true probability of a correct answer on this benchmark, computed from all its recorded generations."],
    ["Cost c", "A provider's true expected cost per query: expected tokens times the per-token price deflated by the margin, in millionths of a dollar."],
    ["Quality gap $\\epsilon_j$", "For an unqualified provider, $\\Theta$ minus its quality. Small gaps take many selections to detect."],
    ["Cost gap $\\Delta_j$", "For a qualified provider other than $i^*$, its cost minus $i^*$'s cost. Small gaps take many selections to resolve."],
    ["$\\beta(m)$", "The quality radius after m selections. The optimistic quality estimate is the empirical accuracy plus $\\beta(m)$."],
    ["$\\rho(m)$", "The cost radius after m selections, $(1 + \\gamma)\\, C_{\\max}\\, \\beta(m)$. The score of a provider is its bid minus $\\rho(m)$."],
    ["$C_{\\max}$", "The payment ceiling and the scale of the cost radius."],
    ["Score", "Bid minus $\\rho(m)$. The mechanism selects the eligible provider with the lowest score."],
    ["Threshold payment", "The highest bid with which the winner would still have won, capped at $C_{\\max}$. It does not depend on the winner's own bid."],
    ["Good event", "The event that every empirical estimate stays within its radius of the truth at every selection count. The guarantees hold on it; with probability at least $1 - \\delta$ it holds."],
    ["$B_{id}$", "The theory's bound on the number of rounds not spent on $i^*$ before it is identified; the sum of the elimination counts of every other provider."],
    ["Collapse", "An episode in which an arm gave more than half of its rounds to one unqualified provider."],
    ["Belief slack", "How far a provider's bid strays from its empirical mean cost, in units of the cost radius. $\\gamma$ is the slack the platform tolerates."],
  ],
};
