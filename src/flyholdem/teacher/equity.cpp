// Original conventional-teacher utility. PokerKit remains the game/settlement
// engine; this evaluator estimates visible-card strength against sampled hands.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>

static uint32_t pack_rank(int category, const int* ranks, int count) {
    uint32_t value = category;
    for (int i = 0; i < 5; ++i) value = value * 15 + (i < count ? ranks[i] : 0);
    return value;
}

static int straight_high(uint32_t bits) {
    for (int high = 14; high >= 6; --high)
        if ((bits & (31u << (high - 4))) == (31u << (high - 4))) return high;
    return (bits & ((1u << 14) | 60u)) == ((1u << 14) | 60u) ? 5 : 0;
}

static uint32_t evaluate(const int32_t* cards, int n) {
    if (n < 5 || n > 7) return 0;
    int count[15] = {}, suits[4] = {};
    uint32_t suit_bits[4] = {}, bits = 0;
    uint64_t seen = 0;
    for (int i = 0; i < n; ++i) {
        const int c = cards[i];
        if (c < 0 || c >= 52 || (seen & (uint64_t(1) << c))) return 0;
        seen |= uint64_t(1) << c;
        const int rank = c / 4 + 2, suit = c % 4;
        ++count[rank]; ++suits[suit]; bits |= 1u << rank; suit_bits[suit] |= 1u << rank;
    }
    int flush = -1;
    for (int s = 0; s < 4; ++s) if (suits[s] >= 5) {
        flush = s; int high = straight_high(suit_bits[s]);
        if (high) return pack_rank(8, &high, 1);
    }
    int ranks[5] = {};
    for (int r = 14; r >= 2; --r) if (count[r] == 4) {
        ranks[0] = r;
        for (int k = 14; k >= 2; --k) if (k != r && count[k]) { ranks[1] = k; break; }
        return pack_rank(7, ranks, 2);
    }
    int trip = 0;
    for (int r = 14; r >= 2; --r) if (count[r] >= 3) { trip = r; break; }
    if (trip) for (int r = 14; r >= 2; --r) if (r != trip && count[r] >= 2) {
        ranks[0] = trip; ranks[1] = r; return pack_rank(6, ranks, 2);
    }
    if (flush >= 0) {
        int j = 0;
        for (int r = 14; r >= 2 && j < 5; --r) if (suit_bits[flush] & (1u << r)) ranks[j++] = r;
        return pack_rank(5, ranks, 5);
    }
    int straight = straight_high(bits);
    if (straight) return pack_rank(4, &straight, 1);
    if (trip) {
        ranks[0] = trip; int j = 1;
        for (int r = 14; r >= 2 && j < 3; --r) if (r != trip && count[r]) ranks[j++] = r;
        return pack_rank(3, ranks, 3);
    }
    int pairs[3] = {}, pair_count = 0;
    for (int r = 14; r >= 2; --r) if (count[r] >= 2) pairs[pair_count++] = r;
    if (pair_count >= 2) {
        ranks[0] = pairs[0]; ranks[1] = pairs[1];
        for (int r = 14; r >= 2; --r) if (r != pairs[0] && r != pairs[1] && count[r]) { ranks[2] = r; break; }
        return pack_rank(2, ranks, 3);
    }
    if (pair_count) {
        ranks[0] = pairs[0]; int j = 1;
        for (int r = 14; r >= 2 && j < 4; --r) if (r != pairs[0] && count[r]) ranks[j++] = r;
        return pack_rank(1, ranks, 4);
    }
    int j = 0;
    for (int r = 14; r >= 2 && j < 5; --r) if (count[r]) ranks[j++] = r;
    return pack_rank(0, ranks, 5);
}

static uint64_t random64(uint64_t& state) {
    uint64_t x = (state += 0x9e3779b97f4a7c15ull);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static int uniform_index(uint64_t& state, int count) {
    const uint64_t bound = uint64_t(count), threshold = -bound % bound;
    uint64_t value;
    do { value = random64(state); } while (value < threshold);
    return int(value % bound);
}

extern "C" void rank_hands(int n, int width, const int32_t* cards, uint32_t* ranks) {
    for (int i = 0; i < n; ++i) ranks[i] = evaluate(cards + i * width, width);
}

extern "C" double visible_equity(const int32_t* hole, const int32_t* board,
                                int board_size, int samples, uint64_t seed) {
    if ((board_size != 0 && board_size != 3 && board_size != 4 && board_size != 5) || samples < 1) return NAN;
    uint64_t known = 0;
    int32_t own[7] = {hole[0],hole[1]}, other[7] = {};
    for (int i = 0; i < 2 + board_size; ++i) {
        int c = i < 2 ? hole[i] : board[i-2];
        if (c < 0 || c >= 52 || (known & (uint64_t(1) << c))) return NAN;
        known |= uint64_t(1) << c;
    }
    for (int i = 0; i < board_size; ++i) own[i+2] = other[i+2] = board[i];
    int pool_template[52], available = 0;
    for (int c = 0; c < 52; ++c) if (!(known & (uint64_t(1) << c))) pool_template[available++] = c;
    int wins_twice = 0;
    for (int repeat = 0; repeat < samples; ++repeat) {
        int pool[52]; std::copy(pool_template, pool_template+available, pool);
        for (int j = 0; j < 7-board_size; ++j) std::swap(pool[j],pool[j+uniform_index(seed,available-j)]);
        other[0] = pool[0]; other[1] = pool[1];
        for (int i = board_size; i < 5; ++i) own[i+2] = other[i+2] = pool[2+i-board_size];
        uint32_t a = evaluate(own,7), b = evaluate(other,7);
        wins_twice += a > b ? 2 : a == b ? 1 : 0;
    }
    return double(wins_twice) / (2*samples);
}
