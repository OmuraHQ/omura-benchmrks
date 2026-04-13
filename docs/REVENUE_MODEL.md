# Omura Revenue Model

## Project Nature: Public Good, Non-Profit Infrastructure

**Omura** is a **public good, non-profit infrastructure project** built for the Walrus protocol ecosystem. All revenue generated is reinvested into:

- **Infrastructure Development**: Maintaining and improving the search infrastructure
- **Protocol Operations**: Server costs, indexing capacity, API hosting
- **Community Rewards**: Distributing value back to content creators and contributors
- **Protocol Sustainability**: Ensuring long-term viability and growth

**No profit extraction**: All revenue flows back into supporting the protocol and rewarding the community.

---

## Finalized Revenue Model: Complete Token Utility Loop

### Core Mechanism: Stake-to-Mint with Priority Indexing Utility

**Complete Flow**:
```
1. STAKE → Stakers stake stablecoins (DAI/USDC/USDT)
         ↓
2. MINT → Earn $OMURA tokens as yield (staking rewards)
         ↓
3. YIELD → DeFi yield from staked stablecoins captured
         ↓
4. SPLIT → Yield split:
           - 50% → Treasury (infrastructure, operations)
           - 50% → Uploader rewards (based on attention/footfall)
         ↓
5. UTILITY → $OMURA tokens used for priority indexing (X-402 payment)
         ↓
6. DEMAND → Priority indexing creates demand for $OMURA tokens
         ↓
7. VALUE → Token demand keeps $OMURA valuable
         ↓
8. LOOP → More value → More stakers → More yield → More rewards → More utility → More demand
```

---

### Complete Token Utility Model

#### **1. Stake-to-Mint (Token Distribution)**
- **Stakers** stake stablecoins (DAI/USDC/USDT) → Earn **$OMURA tokens** as yield
- **Minimum stake**: 100 DAI equivalent
- **Lock periods**: 30/60/90/180/365 days
- **APY (in $OMURA)**: 20% (30d) to 80% (365d) + bonuses
- **Token emission**: Adaptive (50K base + 10K per 10M DAI staked, max 200K/day)

#### **2. DeFi Yield Capture (Public Good Funding)**
- **Staked stablecoins** deployed to Sui DeFi protocols (lending, liquidity pools)
- **DeFi yield captured** (target: 4-6% APY on stablecoins)
- **All yield reinvested** (no profit extraction):
  - **50% → Treasury**: Infrastructure maintenance, operations, protocol development
  - **50% → Uploader Rewards Pool**: Distributed to Walrus uploaders (community benefit)

#### **3. Uploader Rewards (Content Creator Incentives)**
- **50% of DeFi yield** distributed to Walrus protocol data uploaders
- **Distribution based on attention/footfall metrics**:
  - **Footfall (40%)**: Access counts (logarithmic scaling)
  - **Attention (30%)**: Engagement, time spent, interaction rate
  - **Visibility (20%)**: Search impressions, click-through rate
  - **Uniqueness (10%)**: Unique visitors, retention rate
- **Daily distribution**: Rewards calculated and distributed daily
- **Minimum threshold**: 0.01 DAI (prevents dust transactions)

#### **4. Priority Indexing Utility (Token Demand Driver)**
- **$OMURA tokens** used for priority indexing via **X-402 (Payment Required)** format
- **Mechanism**: 
  - Users want their blobs indexed faster
  - Users pay **$OMURA tokens** for priority indexing queue
  - Priority indexing = faster indexing time (e.g., 1 hour vs. 2-3 days)
  - **Fairness guarantee**: Priority indexing only affects indexing speed, NOT search ranking
- **Pricing**: Dynamic based on network load
  - Low load: **0.1 $OMURA per blob**
  - Medium load: **0.5 $OMURA per blob**
  - High load: **1.0 $OMURA per blob** (scales with demand)
- **Payment**: On-chain via Sui smart contract (X-402 payment receipt NFT)

#### **5. Token Demand Loop (Value Mechanism)**
- **Priority indexing** creates demand for $OMURA tokens
- **Token demand** keeps $OMURA valuable (price discovery via DEX trading)
- **More demand** → **Higher price** → **More stakers** → **More yield** → **More rewards**
- **Circular economy**: Value begets value

---

### Complete Revenue Streams Summary

#### **Revenue Stream 1: DeFi Yield (Public Good Funding)**
- **Source**: Yield from staked DAI/USDC/USDT deployed to DeFi
- **Purpose**: Fund infrastructure operations (non-profit)
- **Split**: 
  - 50% → Treasury (infrastructure maintenance, protocol development)
  - 50% → Uploader rewards (community benefit, content creators)
- **Example**: 1M DAI staked @ 5% APY = 50K DAI/year = 25K to treasury (infrastructure) + 25K to uploaders (community)

#### **Revenue Stream 2: Priority Indexing Fees (X-402 Payments)**
- **Source**: Users paying $OMURA tokens for priority indexing
- **Pricing**: 0.1-1.0 $OMURA per blob (dynamic)
- **Purpose**: Fund infrastructure capacity (non-profit)
- **Split** (all reinvested):
  - 50% → Treasury (infrastructure capacity, server costs)
  - 50% → Distributed to stakers (bonus on top of minting rewards, community benefit)
- **Token utility**: Creates demand for $OMURA tokens
- **Fairness**: Only affects indexing speed, not search ranking

#### **Revenue Stream 3: Query Fees (X-402 API Payments)**
- **Source**: Users paying for API access (if quota exceeded)
- **Pricing**: 0.001 $OMURA per query (or free tier with quotas)
- **Purpose**: Fund API infrastructure operations (non-profit)
- **Split** (all reinvested):
  - 50% → Treasury (API server costs, infrastructure maintenance)
  - 50% → Distributed to stakers (bonus, community benefit)
- **Optional**: Can be free tier (1K queries/month) + paid tier (public good access)

---

### Finalized Parameters (COMPLETE)

#### **Staking**
- **Minimum stake**: 100 DAI equivalent (or USDC/USDT/$OMURA)
- **Lock periods**: 30/60/90/180/365 days
- **APY (in $OMURA)**: 
  - 30 days: 20%
  - 60 days: 30%
  - 90 days: 40%
  - 180 days: 60%
  - 365 days: 80%
  - Bonus: +10% if stake ≥ 10,000 DAI
  - Bonus: +20% if stake ≥ 50,000 DAI
  - Bonus: +50% if running indexer node
- **Early unlock penalty**: 50% of earned rewards forfeited

#### **DeFi Yield Split** (All Reinvested - Non-Profit)
- **50% → Treasury**: Infrastructure maintenance, protocol operations, development (public good)
- **50% → Uploader Rewards**: Content creator rewards based on attention/footfall (community benefit)

#### **Uploader Reward Distribution (Daily)**
- **Footfall**: 40% weight (access counts)
- **Attention**: 30% weight (engagement, time spent)
- **Visibility**: 20% weight (search impressions, CTR)
- **Uniqueness**: 10% weight (unique visitors, retention)

#### **Priority Indexing (Token Utility)**
- **Pricing**: 0.1-1.0 $OMURA per blob (dynamic based on demand)
- **Payment method**: X-402 (Payment Required) format via Sui smart contract
- **Fairness**: Only affects indexing speed, NOT search ranking
- **Split** (if applicable): 50% treasury, 50% stakers

#### **Token Emission** (Community Distribution)
- **Total supply**: 1,000,000,000 $OMURA (1 billion)
- **Daily emission**: 50K base + 10K per 10M DAI staked (max 200K/day)
- **Distribution**: 70% stakers (community), 25% indexers (infrastructure providers), 5% treasury (protocol development)

---

### Token Value Mechanism (Complete Loop)

**Step 1: Token Distribution**
- Stakers stake stablecoins → Earn $OMURA tokens (staking rewards)
- Creates initial token supply and distribution

**Step 2: Token Utility (Demand)**
- Users need $OMURA tokens for priority indexing
- Users need $OMURA tokens for API access (if premium tier)
- Creates demand for tokens

**Step 3: Token Value (Price Discovery)**
- Demand → Trading on DEXs (Cetus, Turbos, etc.)
- Price discovery via market forces
- Higher demand → Higher price

**Step 4: Value Loop (Circular Economy)**
- Higher token value → More incentive to stake (earn valuable tokens)
- More stakers → More DeFi yield → More uploader rewards → Better content
- Better content → More users → More priority indexing demand → More token demand
- **Positive feedback loop**: Value begets value

---

### Fairness Guarantees

#### **Search Fairness (Unchanged)**
- ✅ Search ranking based on similarity score ONLY
- ✅ No pay-to-rank mechanisms
- ✅ Priority indexing only affects indexing SPEED, not ranking
- ✅ Uploader rewards based on ORGANIC metrics (footfall, attention), not payment

#### **Indexing Fairness**
- ✅ Priority indexing = faster indexing time (1 hour vs. 2-3 days)
- ✅ Standard indexing = free, just slower
- ✅ Both indexed content ranks the SAME in search results

#### **Reward Fairness**
- ✅ Uploader rewards based on attention/footfall (organic metrics)
- ✅ Logarithmic scaling prevents whale dominance
- ✅ Multiple metrics ensure quality content rewarded more than spam

---

## Final Summary: Complete Revenue Model

**Core Mechanism**: **Stake-to-Mint with Priority Indexing Utility**

1. **Stakers** stake stablecoins → Earn **$OMURA tokens** (staking rewards)
2. **DeFi yield** from staked stablecoins → Split 50/50 (treasury + uploader rewards)
3. **Uploader rewards** → Distributed based on attention/footfall (organic metrics)
4. **$OMURA tokens** → Used for priority indexing (X-402 payment, faster indexing)
5. **Token demand** → Priority indexing creates demand for $OMURA tokens
6. **Token value** → Demand keeps tokens valuable (price discovery via DEX trading)
7. **Value loop** → Higher value → More stakers → More yield → More rewards → More utility → More demand

**Key Features**:
- ✅ **Public good infrastructure** (non-profit, all revenue reinvested)
- ✅ **Fair token distribution** (via staking, not sale)
- ✅ **Infrastructure funding** (DeFi yield → treasury for protocol maintenance)
- ✅ **Content creator incentives** (quality content → more rewards, community benefit)
- ✅ **Token utility** (priority indexing creates demand)
- ✅ **Token value** (demand keeps tokens valuable)
- ✅ **Circular economy** (value begets value)
- ✅ **Search fairness maintained** (rewards based on organic metrics, priority indexing only affects speed, not ranking)
- ✅ **Community-driven** (all revenue flows back to protocol and community)

**Result**: Complete token utility loop for sustainable public good infrastructure, with all revenue reinvested into protocol development, operations, and community rewards. Search fairness maintained while creating sustainable funding model for non-profit infrastructure.
