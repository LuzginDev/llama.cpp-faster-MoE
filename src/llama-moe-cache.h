#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
class  llama_les;

// cache-aware MoE routing ("cache prior", arXiv:2412.00099)
// fills mask[e] with 1.0f for every expert of the layer that is currently in the cache (slots[e] >= 0)
// the caller applies the mask as a bias to the router scores, so that it only changes the ranking of
// the experts and never the weights they are combined with
void llama_moe_cache_prior_mask(const int32_t * slots, int32_t n_expert, float * mask);

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // prior is the lambda of the cache-aware routing, 0 keeps the original expert selection
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float prior = 0.0f);
    ~llama_moe_cache();

    // lambda of the cache-aware routing, 0 = disabled
    float prior() const;

    // true if the experts of the layer il are kept in the cache, so that a cache prior can be applied to its routing
    bool is_cached(int32_t il) const;

    // the slot of each expert of the layer il, -1 if it is not cached, nullptr if the layer is not cached.
    // the slots that the lookahead scheduler reserved for the next token are already in it.
    const int32_t * get_cached_slots(int32_t il) const;

    // the number of experts of the layer il, 0 if the layer is not cached
    int32_t n_expert(int32_t il) const;

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    //
    // lookahead prefetch, used by LES (see llama-les.h)
    //

    // the lookahead scheduler, it observes the selection of each layer and reserves slots ahead of time
    void set_les(llama_les * les);

    // a byte range that fills a reserved slot of a bank
    struct prefetch_copy {
        ggml_backend_t backend;
        ggml_tensor *  dst;      // the device bank
        size_t         dst_offs; // the offset of the reserved slot in the bank
        const void *   src;      // the data of the expert in host memory
        size_t         nbytes;
    };

    // true if the experts of layer il can be prefetched
    bool prefetch_layer(int32_t il) const;

    // device bytes of one expert of layer il, all of its tensors
    size_t prefetch_expert_bytes(int32_t il) const;

    // mark in resident[e] the experts of layer il that are already in the cache
    void prefetch_resident(int32_t il, std::vector<int8_t> & resident) const;

    // reserve a slot for each of the ids that is not cached yet and return the uploads that fill
    // them, at most max_bytes are planned. The reserved slots are not evicted by copy() while the
    // step runs, and the experts are counted as hits by the next copy().
    // returns the number of bytes planned
    size_t prefetch_reserve(int32_t il, const int32_t * ids, size_t n_ids, size_t max_bytes, std::vector<prefetch_copy> & copies);

    // release the reservations of the previous step, called when no graph is in flight
    void prefetch_release();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
