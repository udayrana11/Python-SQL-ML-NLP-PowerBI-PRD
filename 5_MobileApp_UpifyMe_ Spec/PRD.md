# UpifyMe – AI Powered Gamified Career Discovery  
**PRD v1.2** · Owner: **Udaybhan Singh Rana** · Last updated: 2025-05-13

---

## 1. Problem Statement
Students (16 – 20 yrs) struggle to choose careers; existing tools are text-heavy, generic, and non-actionable.

## 2. Why Now?
| Driver | Data Point | Implication |
|--------|-----------|-------------|
| Ed-tech TAM (India) | **$2.8 B**, +25 % CAGR | Large, fast-growing market |
| Gen-Z engagement | 70 % prefer gamified learning apps | Game UX = differentiation |
| AI cost curve | LLM inference ↓ **65 % YoY** | On-device ML now viable |

---

## 3. Objectives & Metrics

| Layer | Metric | Target |
|-------|--------|--------|
| **North-Star** | **Monthly Active Career Explorers (MACE)** | ≥ 50 k by Q4 FY25 |
| Engagement | Task completion | ≥ 60 % |
| Monetisation | Report purchase rate | ≥ 15 % |
| Satisfaction | Report NPS | ≥ 45 |
| Ops | Data latency | ≤ 5 min |

---

## 4. ROI & Cost-of-Delay

| Assumption | Value |
|------------|-------|
| Launch cohort | 50 k MAU |
| Purchase conversion | 15 % |
| ARPPU | ₹399 ($4.80) |
| **MRR @ target** | **$36 k** |
| Team capacity | 3 sprints (2 w ea.) |
| **Cost of delay** | **$6 k / week** |

> *Rationale: every sprint slipped pushes out $6 k in monthly recurring revenue.*

---

## 5. Personas
- **Student Shreya (17):** fun-first, mobile-native, short attention span  
- **Parent Rajesh (42):** accuracy-seeker, pays for reports  
- **Counselor Aditi (34):** classroom facilitator, needs engaging tool

---

## 6. Scope – MVP

| Feature | T-Shirt | In v1 | Notes |
|---------|---------|-------|-------|
| 12 assessment games | L | ✅ | WebGL in Flutter |
| 20 micro-videos | M | ✅ | ≤ 90 s |
| Chatbot (Dialogflow CX) | M | ✅ | 60 intents |
| ML recommendation engine | L | ✅ | Unsupervised + Clustering Algorithm |
| Coin reward system | S | ✅ | Firebase rules |
| Paywall @ 60 % tasks | S | ✅ | Coins or ₹399 or $7.99 |
| Real-time dashboards | M | ✅ | Looker on BigQuery |

*Out of scope v1:* resume builder, offline mode, live counselor chat

---

## 7. User Stories
 Jira link; - will add shortly

---

## 8. UX & Flow Links
- **Low-fi wireframes:** *link here*  - will add shortly
- **High-fi Figma prototype:** *link here* - will add shortly
- **60-sec demo video:** *link here* - will add shortly

---

## 9. Technical Architecture
Frontend → Flutter · Games (WebGL)  
Backend → Firebase Auth + Firestore · Cloud Functions  
ML → AI; retrain weekly  
Analytics → Segment → BigQuery → Looker

**Latency budgets:** < 200 ms API RTT · < 5 min analytics freshness

---

## 10. Dependencies
- Psychometric dataset licence – signed 2025-06-01  
- AI quota upgrade – pro tier requested  
- **Privacy & Compliance:** GDPR & Indian DPDP; under-18 consent flow  
- **Accessibility:** WCAG 2.1 AA; color palette CVD-safe  
- **Localization:** EN-US, EN-IN at launch; Hindi & Spanish Wave 2

---

## 11. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Low early engagement | KPI miss | Med | Coins booster + onboarding AB |
| Model bias | Accuracy | Low | Bias audit, human review loop |
| Play-store rejection | Launch delay | Low | 64-bit libs, privacy policy |
| High infra cost | Margin hit | Med | Auto-scale rules, batch scoring |

---

## 12. Launch Readiness Checklist
- [ ] QA sign-off (crash-free ≥ 99 %)  
- [ ] Legal/compliance sign-off (GDPR, DPDP)  
- [ ] Accessibility audit passed  
- [ ] Payment gateway live & KYC verified  
- [ ] Data-retention policy (90 days) documented  
- [ ] Support macros in Zendesk  
- [ ] Play-store assets uploaded & approved  
- [ ] Exec go/no-go review booked (T-1d)  

---

## 13. Roll-out Plan
1. **Beta – 500 users** (friends & family)  
2. **Wave 1 – India Tier-1 cities** (10 k)  
3. **Wave 2 – US East Coast** (10 k)  
4. **Global GA** once KPIs hit gates

---

## 14. Analytics & Experimentation
- Remote Config splits 50:50  
- Minimum sample size ≈ 12 k sessions (95 % power, +10 % Δ)  
- Looker dashboards email 09:00 IST daily

---

## 15. Appendix
- PRD PDF  
- Jira backlog CSV  
- Data-flow diagram (PNG)  
