/*
 * Copyright (c) 2026 Evolution Technologies Research and Prototype
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The "evo is not responding" issue is likely due to the render loop consuming too much CPU or blocking.
 *
 * sudo dnf install wget
 * wget --version
 * ~/projects/eclipse-workspace-cdt/evox/models$
 * wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
 * hexdump -C mistral-7b-instruct-v0.2.Q4_K_M.gguf | head -n 20
 * hexdump -C mistral-7b-instruct-v0.2.Q4_K_M.bin | head -n 20
 * 
 * Loading file convetible to running animation
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * File: evox/src/main.c
 * Description: EVOX AI Core 5-Axes System with Real-time Neural Monitoring
 *
 * Compilation: gcc -std=c89 -O3 -march=native -mavx2 -mfma -pthread \
 *              -lGL -lGLU -lglut -lSDL2 -lopenal -lssl -lcrypto \
 *              -lmicrohttpd -lmpi -lOpenCL -lm -o evox_core main.c
 *
 * Architecture: AMD Ryzen 5 7520U with Radeon Graphics
 * OS: Linux Fedora 43 (Workstation)
 */

/*=============================================================================
 * SYSTEM HEADERS WITH C89 COMPATIBILITY
 *============================================================================*/

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#define _ISOC99_SOURCE 1

/* C89 Standard Libraries */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <signal.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>

/* POSIX Headers */
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <dirent.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <libgen.h>
#include <regex.h>

/* OpenMPI */
#include <mpi.h>

/* OpenSSL */
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/rand.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/aes.h>

/* libmicrohttpd */
#include <microhttpd.h>

/* OpenGL */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

/* SDL2 */
#include <SDL2/SDL.h>

/* OpenAL */
#include <AL/al.h>
#include <AL/alc.h>

/* OpenCL */
#include <CL/cl.h>

/* AVX-256 Intrinsics */
#include <immintrin.h>

/*=============================================================================
 * NUMA HEADER COMPATIBILITY LAYER
 *============================================================================*/

/* Temporarily disable strict ANSI for NUMA headers */
#ifdef __STRICT_ANSI__
#undef __STRICT_ANSI__
#endif

#ifndef inline
#define inline
#endif

#include <numa.h>
#include <numaif.h>

/*=============================================================================
 * SYSTEM CONSTANTS
 *============================================================================*/

#define EVOX_VERSION_MAJOR             1
#define EVOX_VERSION_MINOR             0
#define EVOX_VERSION_PATCH             0

/* File System */
#define MAX_PATH_LEN                    4096
#define MAX_FILENAME_LEN                256
#define MAX_MODEL_NAME_LEN              128
#define MODELS_DIRECTORY                "./models/"

/* Neural Network Limits */
#define MAX_VOCAB_SIZE                  50000
#define MAX_HIDDEN_SIZE                 4096
#define MAX_LAYERS                       32
#define MAX_HEADS                         32
#define MAX_EXPERTS                       16
#define MAX_NODES                      100000
#define MAX_SYNAPSES                   500000

/* 5-Axes System */
#define AXIS_COUNT                         5
#define AXIS_X_INDEX                       0
#define AXIS_Y_INDEX                       1
#define AXIS_Z_INDEX                       2
#define AXIS_B_INDEX                       3
#define AXIS_R_INDEX                       4

/* Security */
#define KEY_ROTATION_HOURS                28
#define KEY_ROTATION_SECONDS        (28 * 3600)
#define AES_KEY_SIZE                      32
#define SHA256_HASH_SIZE                  32

/* Performance */
#define SIMD_ALIGNMENT                    32
#define CACHE_LINE_SIZE                   64
#define UPDATE_RATE_HZ                    60
#define RENDER_RATE_HZ                     60

/* FSM */
#define FSM_STATES                        14
#define FSM_BOOT_STEPS                     8
#define FSM_EVENTS                         15

/*=============================================================================
 * ENUMERATIONS
 *============================================================================*/

typedef enum {
	FSM_STATE_BOOT = 0,
	FSM_STATE_SELF_TEST,
	FSM_STATE_HARDWARE_INIT,
	FSM_STATE_MODEL_LOAD,
	FSM_STATE_NETWORK_INIT,
	FSM_STATE_CRYPTO_INIT,
	FSM_STATE_RENDERING_INIT,
	FSM_STATE_AUDIO_INIT,
	FSM_STATE_IDLE,
	FSM_STATE_PROCESSING,
	FSM_STATE_LEARNING,
	FSM_STATE_ROUTING,
	FSM_STATE_KEY_ROTATION,
	FSM_STATE_ERROR,
	FSM_STATE_SHUTDOWN
} FSMState;

typedef enum {
	FSM_EVENT_NONE = 0,
	FSM_EVENT_BOOT_COMPLETE,
	FSM_EVENT_BOOT_FAILED,
	FSM_EVENT_MODEL_LOADED,
	FSM_EVENT_MODEL_FAILED,
	FSM_EVENT_NETWORK_READY,
	FSM_EVENT_NETWORK_FAILED,
	FSM_EVENT_KEY_EXPIRING,
	FSM_EVENT_KEY_ROTATED,
	FSM_EVENT_INFERENCE_REQUEST,
	FSM_EVENT_LEARNING_TRIGGER,
	FSM_EVENT_ROUTE_UPDATE,
	FSM_EVENT_ERROR_DETECTED,
	FSM_EVENT_SHUTDOWN_REQUEST
} FSMEvent;

typedef enum {
	MAMDANI_MIN, MAMDANI_MAX, MAMDANI_PROD
} MamdaniType;

/*=============================================================================
 * CORE DATA STRUCTURES
 *============================================================================*/

/* 5-Axes Vector - 32-byte aligned for SIMD */
typedef struct {
	double x; /* Length */
	double y; /* Height */
	double z; /* Width */
	double b; /* Diagonal Base */
	double r; /* Rotation */
} FiveAxisVector;

/* Neural Node for Real-time Monitoring */
typedef struct {
	double activation;
	double membrane_potential;
	double threshold;
	double refractory_period;
	double hebbian_trace;
	double position[AXIS_COUNT];
	unsigned long spike_count;
	double spike_rate;
	double burst_index;
	double phase;
	double color[3];
	double recent_activity[100];
	unsigned int activity_ptr;
	double energy;
	int is_firing;
	double last_spike_time;
} NeuralNode;

/* Synapse for Real-time Monitoring */
typedef struct {
	unsigned int from_node;
	unsigned int to_node;
	double weight;
	double delay;
	double plasticity;
	double luminescence;
	unsigned int firing_count;
	double last_update;
	double transmission_rate;
	double facilitation;
	double color[4];
	int is_active;
	double last_activity;
	double strength_history[100];
	unsigned int history_ptr;
} Synapse;

/* Neural Network with Real-time Monitoring */
typedef struct {
	/* Topology */
	unsigned int num_nodes;
	unsigned int num_synapses;
	unsigned int vocab_size;
	unsigned int hidden_size;
	unsigned int num_layers;
	unsigned int num_heads;
	unsigned int num_experts;

	/* Data Arrays */
	NeuralNode *nodes;
	Synapse *synapses;
	double *node_activations;
	double *node_deltas;
	double *layer_outputs;
	double *attention_weights;
	unsigned int *expert_routing;
	double *expert_gates;

	/* Real-time Metrics */
	double current_activity;
	double peak_activity;
	double avg_activity;
	double firing_rate;
	double sync_index;
	double oscillation_freq;
	unsigned long total_spikes;
	unsigned int active_neurons;
	unsigned int bursting_neurons;
	double *activity_history;
	unsigned int history_size;
	unsigned int history_pos;
	double *power_spectrum;
	unsigned int spectrum_size;

	/* Synchronization */
	pthread_spinlock_t network_lock;
	pthread_rwlock_t monitor_lock;
	int is_allocated;
	unsigned long update_count;
} NeuralNetwork;

/* Model Information */
typedef struct {
	char filename[MAX_FILENAME_LEN];
	char base_name[MAX_MODEL_NAME_LEN];
	char size_label[64];
	char fine_tune[MAX_MODEL_NAME_LEN];
	char version[32];
	char encoding[32];
	char type[32];
	int shard_num;
	int shard_total;
	unsigned char checksum[SHA256_HASH_SIZE];
	size_t file_size;
	int is_validated;

	/* Model Parameters */
	unsigned int vocab_size;
	unsigned int hidden_size;
	unsigned int num_layers;
	unsigned int num_heads;
	unsigned int num_experts;
	unsigned int total_parameters;
	unsigned int num_nodes;
	unsigned int num_synapses;
} ModelInfo;

/* Security Context */
typedef struct {
	EVP_CIPHER_CTX *cipher_ctx;
	EVP_MD_CTX *md_ctx;
	EVP_PKEY *pkey;
	unsigned char aes_key[AES_KEY_SIZE];
	unsigned char aes_iv[16];
	unsigned long key_creation;
	unsigned long key_expiry;
	unsigned int rotations;
	pthread_mutex_t crypto_mutex;
	int is_allocated;
} SecurityContext;

/* OpenGL Context */
typedef struct {
	SDL_Window *window;
	SDL_GLContext gl_context;
	int width;
	int height;
	FiveAxisVector camera_pos;
	double rotation;
	pthread_mutex_t render_mutex;
	int is_allocated;
} OpenGLContext;

/* Boot Sequence */
typedef struct {
	unsigned int step;
	unsigned int completed;
	unsigned int errors;
	unsigned int boot_complete;
	unsigned int boot_successful;
	double timings[FSM_BOOT_STEPS];
	unsigned long start_time;
	char error_msg[256];
} BootSequence;

/* FSM Transition */
typedef struct {
	FSMState current;
	FSMState previous;
	unsigned long entry_time;
	FSMEvent last_event;
	pthread_mutex_t fsm_mutex;
	pthread_cond_t fsm_cond;
} FSMContext;

/* Performance Metrics */
typedef struct {
	double fps;
	double frame_time;
	double cpu_usage;
	double memory_usage;
	double inference_latency;
	unsigned long total_frames;
	unsigned long total_inferences;
	struct timeval start_time;
} PerformanceMetrics;

/* Main System Structure */
typedef struct EVOXCoreSystem {
	/* Version */
	unsigned int version_major;
	unsigned int version_minor;
	unsigned int version_patch;

	/* State */
	FSMContext fsm;
	BootSequence boot;
	volatile sig_atomic_t shutdown;
	unsigned int error_code;

	/* 5-Axes */
	FiveAxisVector axes[AXIS_COUNT];
	FiveAxisVector origin;
	double axis_weights[AXIS_COUNT];

	/* Neural Network */
	NeuralNetwork *network;
	ModelInfo model_info;
	unsigned int model_loaded;

	/* Security */
	SecurityContext *crypto;

	/* Rendering */
	OpenGLContext *gl;
	int rendering_enabled;

	/* Audio */
	ALCdevice *audio_device;
	ALCcontext *audio_context;
	int audio_enabled;

	/* Performance */
	PerformanceMetrics perf;

	/* Synchronization */
	pthread_mutex_t system_mutex;
	pthread_rwlock_t model_lock;
	pthread_spinlock_t perf_lock;

	/* Runtime */
	unsigned long start_time;
	unsigned long update_count;
} EVOXCoreSystem;

/*=============================================================================
 * FUNCTION PROTOTYPES
 *============================================================================*/

/* Utility */
static double get_time(void);
static void* aligned_alloc_wrap(size_t size, size_t alignment);
static void secure_zero(void *ptr, size_t size);

/* 5-Axes */
static void five_axes_init(EVOXCoreSystem *system);
static double five_axes_weight(const FiveAxisVector *point,
		const double *weights);

/* Security */
static SecurityContext* security_init(void);
static int security_rotate_keys(SecurityContext *sec);
static int security_check_expiry(SecurityContext *sec, unsigned long now);

/* Model Loading */
static int model_load_bin(EVOXCoreSystem *system, const char *path);
static int model_scan_and_load(EVOXCoreSystem *system);
static int gguf_to_bin_convert(EVOXCoreSystem *system, const char *gguf_path);

/* Neural Processing */
static void neural_init_network(NeuralNetwork *net, unsigned int nodes,
		unsigned int synapses);
static void neural_update_real_time(EVOXCoreSystem *system);
static void neural_compute_metrics(NeuralNetwork *net);

/* Rendering */
static OpenGLContext* rendering_init(int width, int height);
static void rendering_render(EVOXCoreSystem *system);
static void rendering_render_axes(EVOXCoreSystem *system);
static void rendering_render_network(EVOXCoreSystem *system);
static void rendering_render_metrics(EVOXCoreSystem *system);

/* FSM */
static void fsm_init(FSMContext *fsm);
static void fsm_process(FSMContext *fsm, FSMEvent event, unsigned long time);

/* Boot */
static void boot_init(BootSequence *boot);
static int boot_step(BootSequence *boot, EVOXCoreSystem *system);
static void boot_execute(EVOXCoreSystem *system);

/* Audio */
static int audio_init(EVOXCoreSystem *system);
static void audio_play_spike(EVOXCoreSystem *system, const double *pos,
		double intensity);

/* Main Loop */
static void main_loop(EVOXCoreSystem *system);

/* Cleanup */
static void system_cleanup(EVOXCoreSystem *system);

/*=============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

static double get_time(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
}

static void* aligned_alloc_wrap(size_t size, size_t alignment) {
	void *ptr = NULL;
	if (posix_memalign(&ptr, alignment, size) != 0)
		return NULL;
	memset(ptr, 0, size);
	return ptr;
}

static void secure_zero(void *ptr, size_t size) {
	volatile unsigned char *p = (volatile unsigned char*) ptr;
	while (size--)
		*p++ = 0;
}

/*=============================================================================
 * 5-AXES MATHEMATICS
 *============================================================================*/

static void five_axes_init(EVOXCoreSystem *system) {
	double inv_sqrt3 = 1.0 / sqrt(3.0);

	/* Origin */
	system->origin.x = 0.0;
	system->origin.y = 0.0;
	system->origin.z = 0.0;
	system->origin.b = 0.0;
	system->origin.r = 0.0;

	/* X Axis - Length */
	system->axes[AXIS_X_INDEX].x = 1.0;
	system->axes[AXIS_X_INDEX].y = 0.0;
	system->axes[AXIS_X_INDEX].z = 0.0;
	system->axes[AXIS_X_INDEX].b = 0.0;
	system->axes[AXIS_X_INDEX].r = 0.0;

	/* Y Axis - Height */
	system->axes[AXIS_Y_INDEX].x = 0.0;
	system->axes[AXIS_Y_INDEX].y = 1.0;
	system->axes[AXIS_Y_INDEX].z = 0.0;
	system->axes[AXIS_Y_INDEX].b = 0.0;
	system->axes[AXIS_Y_INDEX].r = 0.0;

	/* Z Axis - Width */
	system->axes[AXIS_Z_INDEX].x = 0.0;
	system->axes[AXIS_Z_INDEX].y = 0.0;
	system->axes[AXIS_Z_INDEX].z = 1.0;
	system->axes[AXIS_Z_INDEX].b = 0.0;
	system->axes[AXIS_Z_INDEX].r = 0.0;

	/* B Axis - Diagonal */
	system->axes[AXIS_B_INDEX].x = inv_sqrt3;
	system->axes[AXIS_B_INDEX].y = inv_sqrt3;
	system->axes[AXIS_B_INDEX].z = inv_sqrt3;
	system->axes[AXIS_B_INDEX].b = 1.0;
	system->axes[AXIS_B_INDEX].r = 0.0;

	/* R Axis - Rotation */
	system->axes[AXIS_R_INDEX].x = 0.0;
	system->axes[AXIS_R_INDEX].y = 0.0;
	system->axes[AXIS_R_INDEX].z = 0.0;
	system->axes[AXIS_R_INDEX].b = 0.0;
	system->axes[AXIS_R_INDEX].r = 1.0;

	/* Weights */
	system->axis_weights[0] = 0.33;
	system->axis_weights[1] = 0.34;
	system->axis_weights[2] = 0.33;
	system->axis_weights[3] = 0.0;
	system->axis_weights[4] = 0.0;
}

static double five_axes_weight(const FiveAxisVector *point,
		const double *weights) {
	double x = point->x, y = point->y, z = point->z, b = point->b, r = point->r;
	double dist2 = x * x + y * y + z * z + b * b + r * r;
	double w_origin = exp(-sqrt(dist2));

	double w_pos = (fmax(x, 0.0) + fmax(y, 0.0) + fmax(z, 0.0) + fmax(b, 0.0)
			+ fmax(r, 0.0)) / 5.0;

	double w_neg = (fmax(-x, 0.0) + fmax(-y, 0.0) + fmax(-z, 0.0)
			+ fmax(-b, 0.0) + fmax(-r, 0.0)) / 5.0;

	return weights[0] * w_origin + weights[1] * w_pos + weights[2] * w_neg;
}

/*=============================================================================
 * SECURITY IMPLEMENTATION
 *============================================================================*/

static SecurityContext* security_init(void) {
	SecurityContext *sec = (SecurityContext*) calloc(1,
			sizeof(SecurityContext));
	if (!sec)
		return NULL;

	OpenSSL_add_all_algorithms();
	ERR_load_crypto_strings();

	sec->cipher_ctx = EVP_CIPHER_CTX_new();
	sec->md_ctx = EVP_MD_CTX_new();

	RAND_bytes(sec->aes_key, AES_KEY_SIZE);
	RAND_bytes(sec->aes_iv, 16);

	sec->key_creation = (unsigned long) time(NULL);
	sec->key_expiry = sec->key_creation + KEY_ROTATION_SECONDS;
	sec->rotations = 0;

	pthread_mutex_init(&sec->crypto_mutex, NULL);
	sec->is_allocated = 1;

	printf("[SECURITY] Context initialized\n");
	return sec;
}

static int security_rotate_keys(SecurityContext *sec) {
	pthread_mutex_lock(&sec->crypto_mutex);
	RAND_bytes(sec->aes_key, AES_KEY_SIZE);
	RAND_bytes(sec->aes_iv, 16);
	sec->key_creation = (unsigned long) time(NULL);
	sec->key_expiry = sec->key_creation + KEY_ROTATION_SECONDS;
	sec->rotations++;
	pthread_mutex_unlock(&sec->crypto_mutex);
	return 1;
}

static int security_check_expiry(SecurityContext *sec, unsigned long now) {
	return (now >= sec->key_expiry) ? 1 : 0;
}

/*=============================================================================
 * NEURAL NETWORK INITIALIZATION
 *============================================================================*/

static void neural_init_network(NeuralNetwork *net, unsigned int nodes,
		unsigned int synapses) {
	unsigned int i;

	net->num_nodes = nodes;
	net->num_synapses = synapses;
	net->vocab_size = 32000;
	net->hidden_size = 4096;
	net->num_layers = 32;
	net->num_heads = 32;
	net->num_experts = 8;

	/* Allocate arrays */
	net->nodes = (NeuralNode*) calloc(nodes, sizeof(NeuralNode));
	net->synapses = (Synapse*) calloc(synapses, sizeof(Synapse));
	net->node_activations = (double*) calloc(nodes, sizeof(double));
	net->node_deltas = (double*) calloc(nodes, sizeof(double));
	net->layer_outputs = (double*) calloc(nodes, sizeof(double));
	net->attention_weights = (double*) calloc(nodes * nodes, sizeof(double));
	net->expert_routing = (unsigned int*) calloc(MAX_EXPERTS,
			sizeof(unsigned int));
	net->expert_gates = (double*) calloc(MAX_EXPERTS, sizeof(double));

	/* Metrics buffers */
	net->history_size = 1000;
	net->spectrum_size = 256;
	net->activity_history = (double*) calloc(1000, sizeof(double));
	net->power_spectrum = (double*) calloc(256, sizeof(double));

	/* Initialize nodes */
	for (i = 0; i < nodes; i++) {
		NeuralNode *node = &net->nodes[i];
		node->activation = ((double) rand() / RAND_MAX) * 0.1;
		node->membrane_potential = -70.0 + ((double) rand() / RAND_MAX) * 10.0;
		node->threshold = -55.0;
		node->refractory_period = 0.002;
		node->hebbian_trace = 0.0;
		node->spike_count = 0;
		node->spike_rate = 0.0;
		node->burst_index = 0.0;
		node->phase = ((double) rand() / RAND_MAX) * 2.0 * M_PI;
		node->energy = 0.0;
		node->is_firing = 0;
		node->last_spike_time = 0.0;
		node->activity_ptr = 0;

		/* Position in 5-axes space */
		node->position[0] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[1] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[2] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[3] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[4] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;

		/* Color based on position */
		node->color[0] = 0.5 + 0.5 * sin(node->position[0] * M_PI);
		node->color[1] = 0.5 + 0.5 * sin(node->position[1] * M_PI + 2.0);
		node->color[2] = 0.5 + 0.5 * sin(node->position[2] * M_PI + 4.0);
	}

	/* Initialize synapses */
	for (i = 0; i < synapses; i++) {
		Synapse *syn = &net->synapses[i];
		syn->from_node = rand() % nodes;
		syn->to_node = rand() % nodes;
		syn->weight = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		syn->delay = ((double) rand() / RAND_MAX) * 0.01;
		syn->plasticity = 0.01;
		syn->luminescence = ((double) rand() / RAND_MAX) * 0.3;
		syn->firing_count = 0;
		syn->transmission_rate = 1.0;
		syn->facilitation = 0.0;
		syn->is_active = 0;
		syn->last_activity = 0.0;
		syn->history_ptr = 0;

		syn->color[0] = 0.5 + 0.5 * syn->weight;
		syn->color[1] = 0.2;
		syn->color[2] = 0.5 - 0.5 * syn->weight;
		syn->color[3] = 0.3;
	}

	/* Initialize locks */
	pthread_spin_init(&net->network_lock, PTHREAD_PROCESS_PRIVATE);
	pthread_rwlock_init(&net->monitor_lock, NULL);

	net->is_allocated = 1;
	net->update_count = 0;

	printf("[NEURAL] Network initialized: %u nodes, %u synapses\n", nodes,
			synapses);
}

/*=============================================================================
 * REAL-TIME NEURAL UPDATE
 *============================================================================*/

static void neural_update_real_time(EVOXCoreSystem *system) {
	NeuralNetwork *net = system->network;
	unsigned int i, j;
	double time = get_time();
	double sum_act = 0.0, sum_sq = 0.0;
	unsigned int active = 0, bursting = 0;
	double phase_sin = 0.0, phase_cos = 0.0;

	if (!net || !net->is_allocated)
		return;

	pthread_spin_lock(&net->network_lock);
	pthread_rwlock_wrlock(&net->monitor_lock);

	/* Update each node */
	for (i = 0; i < net->num_nodes; i++) {
		NeuralNode *node = &net->nodes[i];
		double input = 0.0;

		/* Integrate synaptic inputs */
		for (j = 0; j < net->num_synapses; j++) {
			Synapse *syn = &net->synapses[j];
			if (syn->to_node == i) {
				NeuralNode *pre = &net->nodes[syn->from_node];
				input += pre->activation * syn->weight;

				if (pre->is_firing) {
					syn->is_active = 1;
					syn->last_activity = time;
					syn->firing_count++;
					syn->luminescence += 0.1;
					if (syn->luminescence > 1.0)
						syn->luminescence = 1.0;
				}
			}
		}

		/* Leaky integrate-and-fire */
		double leak = 0.01 * node->membrane_potential;
		node->membrane_potential += input - leak;

		/* Spike detection */
		if (node->membrane_potential >= node->threshold
				&& (time - node->last_spike_time) > node->refractory_period) {

			node->membrane_potential = -70.0;
			node->last_spike_time = time;
			node->spike_count++;
			node->is_firing = 1;

			/* Update spike rate */
			node->spike_rate = node->spike_rate * 0.95
					+ 0.05 / node->refractory_period;

			/* Burst detection */
			if (node->spike_rate > 100.0) {
				node->burst_index += 0.1;
				if (node->burst_index > 1.0)
					node->burst_index = 1.0;
			}

			node->energy = 100.0;
		} else {
			node->is_firing = 0;
			node->energy *= 0.95;
		}

		/* Activation function */
		node->activation = 1.0 / (1.0 + exp(-node->membrane_potential / 10.0));

		/* Phase update */
		node->phase += 0.1 * node->activation;
		if (node->phase > 2.0 * M_PI)
			node->phase -= 2.0 * M_PI;

		/* Store recent activity */
		node->recent_activity[node->activity_ptr % 100] = node->activation;
		node->activity_ptr++;

		/* Statistics */
		sum_act += node->activation;
		sum_sq += node->activation * node->activation;
		if (node->activation > 0.1)
			active++;
		if (node->burst_index > 0.5)
			bursting++;
		phase_sin += sin(node->phase);
		phase_cos += cos(node->phase);
	}

	/* Compute network metrics */
	net->current_activity = sum_act / net->num_nodes;
	net->avg_activity = net->avg_activity * 0.99 + net->current_activity * 0.01;
	if (net->current_activity > net->peak_activity) {
		net->peak_activity = net->current_activity;
	}

	net->firing_rate = net->current_activity * 1000.0;
	net->sync_index = sqrt(phase_sin * phase_sin + phase_cos * phase_cos)
			/ net->num_nodes;
	net->oscillation_freq = 1.0 + net->sync_index * 9.0;
	net->active_neurons = active;
	net->bursting_neurons = bursting;
	net->total_spikes += active;

	/* Update history */
	net->activity_history[net->history_pos % net->history_size] =
			net->current_activity;
	net->history_pos++;

	/* Simple power spectrum */
	for (i = 0; i < net->spectrum_size; i++) {
		double freq = (double) i / net->spectrum_size * 50.0;
		net->power_spectrum[i] = exp(-freq * freq / 100.0)
				* net->current_activity * 10.0;
	}

	net->update_count++;

	pthread_rwlock_unlock(&net->monitor_lock);
	pthread_spin_unlock(&net->network_lock);
}

static void neural_compute_metrics(NeuralNetwork *net) {
	/* Metrics are updated in neural_update_real_time */
	(void) net;
}

/*=============================================================================
 * MODEL LOADING
 *============================================================================*/

static int model_load_bin(EVOXCoreSystem *system, const char *path) {
	FILE *fp;
	unsigned char header[256];
	unsigned int nodes, synapses;

	printf("[MODEL] Loading: %s\n", path);

	fp = fopen(path, "rb");
	if (!fp) {
		printf("[ERROR] Cannot open file\n");
		return -1;
	}

	/* Read header */
	if (fread(header, 1, sizeof(header), fp) != sizeof(header)) {
		printf("[ERROR] Invalid header\n");
		fclose(fp);
		return -1;
	}

	/* Verify EVOX magic */
	if (header[0] == 'E' && header[1] == 'V' && header[2] == 'O'
			&& header[3] == 'X') {
		printf("[MODEL] EVOX format v%d.%d\n", header[4], header[5]);
	}

	/* Read topology */
	fseek(fp, 256, SEEK_SET);
	fread(&nodes, sizeof(nodes), 1, fp);
	fread(&synapses, sizeof(synapses), 1, fp);

	printf("[MODEL] Topology: %u nodes, %u synapses\n", nodes, synapses);

	/* Initialize network */
	if (!system->network) {
		system->network = (NeuralNetwork*) calloc(1, sizeof(NeuralNetwork));
		neural_init_network(system->network, nodes, synapses);
	}

	/* Read node data */
	if (system->network->nodes) {
		unsigned int i;
		for (i = 0; i < nodes && i < system->network->num_nodes; i++) {
			NeuralNode *node = &system->network->nodes[i];
			fread(node->position, sizeof(double), AXIS_COUNT, fp);
			fread(&node->threshold, sizeof(double), 1, fp);
		}
	}

	/* Read synapse data */
	if (system->network->synapses) {
		unsigned int i;
		for (i = 0; i < synapses && i < system->network->num_synapses; i++) {
			Synapse *syn = &system->network->synapses[i];
			fread(&syn->from_node, sizeof(unsigned int), 1, fp);
			fread(&syn->to_node, sizeof(unsigned int), 1, fp);
			fread(&syn->weight, sizeof(double), 1, fp);
			fread(&syn->delay, sizeof(double), 1, fp);
		}
	}

	fclose(fp);

	system->model_loaded = 1;
	strncpy(system->model_info.filename, path, MAX_FILENAME_LEN - 1);

	printf("[MODEL] Loaded successfully\n");
	return 0;
}

static int model_scan_and_load(EVOXCoreSystem *system) {
	DIR *dir;
	struct dirent *entry;

	dir = opendir(MODELS_DIRECTORY);
	if (!dir) {
		mkdir(MODELS_DIRECTORY, 0755);
		return 0;
	}

	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_name[0] == '.')
			continue;

		const char *ext = strrchr(entry->d_name, '.');
		if (ext && strcmp(ext, ".bin") == 0) {
			char path[MAX_PATH_LEN];
			snprintf(path, sizeof(path), "%s%s", MODELS_DIRECTORY,
					entry->d_name);

			if (model_load_bin(system, path) == 0) {
				closedir(dir);
				return 1;
			}
		}
	}

	closedir(dir);
	return 0;
}

/*=============================================================================
 * OPENGL RENDERING
 *============================================================================*/

static OpenGLContext* rendering_init(int width, int height) {
	OpenGLContext *gl = (OpenGLContext*) calloc(1, sizeof(OpenGLContext));
	if (!gl)
		return NULL;

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		free(gl);
		return NULL;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	gl->window = SDL_CreateWindow("EVOX AI Core - 5-Axes Neural Visualization",
	SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height,
			SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

	if (!gl->window) {
		SDL_Quit();
		free(gl);
		return NULL;
	}

	gl->gl_context = SDL_GL_CreateContext(gl->window);
	if (!gl->gl_context) {
		SDL_DestroyWindow(gl->window);
		SDL_Quit();
		free(gl);
		return NULL;
	}

	SDL_GL_SetSwapInterval(1);

	gl->width = width;
	gl->height = height;

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) width / height, 0.1, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gl->camera_pos.x = 5.0;
	gl->camera_pos.y = 5.0;
	gl->camera_pos.z = 10.0;
	gl->camera_pos.b = 0.0;
	gl->camera_pos.r = 0.0;
	gl->rotation = 0.0;

	pthread_mutex_init(&gl->render_mutex, NULL);
	gl->is_allocated = 1;

	printf("[RENDER] OpenGL initialized (%dx%d)\n", width, height);
	return gl;
}

static void rendering_render_axes(EVOXCoreSystem *system) {
	OpenGLContext *gl = system->gl;

	glPushMatrix();
	glRotated(gl->rotation, 0.0, 1.0, 0.0);

	/* X Axis - Red */
	glLineWidth(3.0);
	glBegin(GL_LINES);
	glColor4f(1.0f, 0.0f, 0.0f, 0.8f);
	glVertex3f(-3.0f, 0.0f, 0.0f);
	glVertex3f(3.0f, 0.0f, 0.0f);
	glEnd();

	/* Y Axis - Green */
	glBegin(GL_LINES);
	glColor4f(0.0f, 1.0f, 0.0f, 0.8f);
	glVertex3f(0.0f, -3.0f, 0.0f);
	glVertex3f(0.0f, 3.0f, 0.0f);
	glEnd();

	/* Z Axis - Blue */
	glBegin(GL_LINES);
	glColor4f(0.0f, 0.0f, 1.0f, 0.8f);
	glVertex3f(0.0f, 0.0f, -3.0f);
	glVertex3f(0.0f, 0.0f, 3.0f);
	glEnd();

	/* B Axis - Purple */
	glBegin(GL_LINES);
	glColor4f(0.5f, 0.0f, 0.5f, 0.6f);
	glVertex3f(-2.5f, -2.5f, -2.5f);
	glVertex3f(2.5f, 2.5f, 2.5f);
	glEnd();

	/* R Axis origin - Yellow */
	glPointSize(12.0f);
	glBegin(GL_POINTS);
	glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glEnd();

	glPopMatrix();
}

static void rendering_render_network(EVOXCoreSystem *system) {
	NeuralNetwork *net = system->network;
	unsigned int i;
	double time = get_time();

	if (!net || !net->is_allocated)
		return;

	pthread_rwlock_rdlock(&net->monitor_lock);

	/* Draw synapses */
	glLineWidth(1.0f);
	glBegin(GL_LINES);

	for (i = 0; i < net->num_synapses; i += 5) {
		Synapse *syn = &net->synapses[i];
		NeuralNode *from = &net->nodes[syn->from_node];
		NeuralNode *to = &net->nodes[syn->to_node];

		float alpha = 0.2f + 0.3f * (float) syn->luminescence;
		if (syn->is_active && (time - syn->last_activity) < 0.1) {
			alpha = 1.0f;
		}

		glColor4f((float) syn->color[0], (float) syn->color[1],
				(float) syn->color[2], alpha);

		glVertex3f((float) from->position[0] * 2.0f,
				(float) from->position[1] * 2.0f,
				(float) from->position[2] * 2.0f);
		glVertex3f((float) to->position[0] * 2.0f,
				(float) to->position[1] * 2.0f, (float) to->position[2] * 2.0f);
	}
	glEnd();

	/* Draw nodes */
	glPointSize(2.0f);
	glBegin(GL_POINTS);

	for (i = 0; i < net->num_nodes; i++) {
		NeuralNode *node = &net->nodes[i];
		float size, alpha;

		if (node->is_firing) {
			size = 8.0f + 4.0f * (float) node->activation;
			alpha = 1.0f;
		} else {
			size = 3.0f + 2.0f * (float) node->activation;
			alpha = 0.5f + 0.3f * (float) node->activation;
		}

		glPointSize(size);

		if (node->burst_index > 0.5) {
			glColor4f(1.0f, 0.3f, 0.0f, alpha);
		} else if (node->spike_rate > 50.0) {
			glColor4f(1.0f, 1.0f, 0.0f, alpha);
		} else {
			glColor4f((float) node->color[0] * (float) node->activation,
					(float) node->color[1] * (float) node->activation,
					(float) node->color[2] * (float) node->activation, alpha);
		}

		glVertex3f((float) node->position[0] * 2.0f,
				(float) node->position[1] * 2.0f,
				(float) node->position[2] * 2.0f);
	}
	glEnd();

	pthread_rwlock_unlock(&net->monitor_lock);
}

static void rendering_render_metrics(EVOXCoreSystem *system) {
	NeuralNetwork *net = system->network;
	OpenGLContext *gl = system->gl;
	char text[256];
	int y = 20;
	unsigned int i;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, gl->width, gl->height, 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);

	/* Model info */
	snprintf(text, sizeof(text), "EVOX AI Core v%u.%u.%u",
			system->version_major, system->version_minor,
			system->version_patch);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glRasterPos2i(10, y);
	for (i = 0; text[i]; i++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
	y += 20;

	if (net && net->is_allocated) {
		/* Activity */
		snprintf(text, sizeof(text), "Activity: %.1f%% (Peak: %.1f%%)",
				net->current_activity * 100.0, net->peak_activity * 100.0);
		glRasterPos2i(10, y);
		if (net->current_activity > 0.7)
			glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
		else if (net->current_activity > 0.4)
			glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
		else
			glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
		for (i = 0; text[i]; i++)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
		y += 20;

		/* Firing rate */
		snprintf(text, sizeof(text), "Firing: %.1f Hz | Spikes: %lu",
				net->firing_rate, net->total_spikes);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		glRasterPos2i(10, y);
		for (i = 0; text[i]; i++)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
		y += 20;

		/* Synchronization */
		snprintf(text, sizeof(text), "Sync: %.2f | Freq: %.1f Hz",
				net->sync_index, net->oscillation_freq);
		glRasterPos2i(10, y);
		for (i = 0; text[i]; i++)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
		y += 20;

		/* Active neurons */
		snprintf(text, sizeof(text), "Active: %u/%u | Bursting: %u",
				net->active_neurons, net->num_nodes, net->bursting_neurons);
		glRasterPos2i(10, y);
		for (i = 0; text[i]; i++)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
		y += 20;

		/* Model info */
		snprintf(text, sizeof(text), "Model: %s", system->model_info.filename);
		glRasterPos2i(10, y);
		for (i = 0; text[i]; i++)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
		y += 20;

		/* Power spectrum (simplified bar graph) */
		int bar_x = 10;
		int bar_y = y + 20;
		int bar_width = 5;
		int bar_spacing = 2;

		for (i = 0; i < 50 && i < net->spectrum_size; i++) {
			float power = (float) net->power_spectrum[i] / 100.0f;
			int height = (int) (power * 40);

			glBegin(GL_QUADS);
			glColor4f(0.0f, power, 1.0f - power, 0.8f);
			glVertex2i(bar_x + i * (bar_width + bar_spacing), bar_y);
			glVertex2i(bar_x + i * (bar_width + bar_spacing) + bar_width,
					bar_y);
			glVertex2i(bar_x + i * (bar_width + bar_spacing) + bar_width,
					bar_y + height);
			glVertex2i(bar_x + i * (bar_width + bar_spacing), bar_y + height);
			glEnd();
		}
	} else {
		snprintf(text, sizeof(text),
				"No model loaded - waiting for BIN files in ./models/");
		glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
		glRasterPos2i(10, y);
		for (i = 0; text[i]; i++)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
	}

	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

static void rendering_render(EVOXCoreSystem *system) {
	OpenGLContext *gl = system->gl;

	if (!gl || !gl->is_allocated)
		return;

	pthread_mutex_lock(&gl->render_mutex);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	/* Set camera */
	gluLookAt(gl->camera_pos.x, gl->camera_pos.y, gl->camera_pos.z, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0);

	/* Apply auto-rotation */
	gl->rotation += 0.5;
	if (gl->rotation > 360.0)
		gl->rotation -= 360.0;

	/* Render 5-axes */
	rendering_render_axes(system);

	/* Render neural network if loaded */
	if (system->model_loaded && system->network) {
		rendering_render_network(system);
	}

	/* Render metrics overlay */
	rendering_render_metrics(system);

	SDL_GL_SwapWindow(gl->window);

	system->perf.total_frames++;

	pthread_mutex_unlock(&gl->render_mutex);
}

/*=============================================================================
 * AUDIO SYSTEM
 *============================================================================*/

static int audio_init(EVOXCoreSystem *system) {
	system->audio_device = alcOpenDevice(NULL);
	if (!system->audio_device)
		return 0;

	system->audio_context = alcCreateContext(system->audio_device, NULL);
	if (!system->audio_context) {
		alcCloseDevice(system->audio_device);
		return 0;
	}

	alcMakeContextCurrent(system->audio_context);
	system->audio_enabled = 1;

	printf("[AUDIO] Initialized\n");
	return 1;
}

static void audio_play_spike(EVOXCoreSystem *system, const double *pos,
		double intensity) {
	/* Audio playback implementation - simplified */
	(void) system;
	(void) pos;
	(void) intensity;
}

/*=============================================================================
 * FINITE STATE MACHINE
 *============================================================================*/

static const FSMState fsm_table[FSM_STATES][FSM_EVENTS] =
		{
		/* BOOT */{ FSM_STATE_BOOT, FSM_STATE_SELF_TEST, FSM_STATE_ERROR },
		/* SELF_TEST */{ FSM_STATE_SELF_TEST, FSM_STATE_HARDWARE_INIT,
				FSM_STATE_ERROR },
		/* HARDWARE_INIT */{ FSM_STATE_HARDWARE_INIT, FSM_STATE_MODEL_LOAD,
				FSM_STATE_ERROR },
		/* MODEL_LOAD */{ FSM_STATE_MODEL_LOAD, FSM_STATE_NETWORK_INIT,
				FSM_STATE_ERROR },
		/* NETWORK_INIT */{ FSM_STATE_NETWORK_INIT, FSM_STATE_CRYPTO_INIT,
				FSM_STATE_ERROR },
		/* CRYPTO_INIT */{ FSM_STATE_CRYPTO_INIT, FSM_STATE_RENDERING_INIT,
				FSM_STATE_ERROR },
		/* RENDERING_INIT */{ FSM_STATE_RENDERING_INIT, FSM_STATE_AUDIO_INIT,
				FSM_STATE_ERROR },
				/* AUDIO_INIT */{ FSM_STATE_AUDIO_INIT, FSM_STATE_IDLE,
						FSM_STATE_ERROR },
				/* IDLE */{ FSM_STATE_IDLE, FSM_STATE_IDLE, FSM_STATE_IDLE,
						FSM_STATE_IDLE, FSM_STATE_IDLE, FSM_STATE_IDLE,
						FSM_STATE_IDLE, FSM_STATE_KEY_ROTATION, FSM_STATE_IDLE,
						FSM_STATE_PROCESSING, FSM_STATE_LEARNING,
						FSM_STATE_ROUTING, FSM_STATE_KEY_ROTATION,
						FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },
				/* PROCESSING */{ FSM_STATE_PROCESSING, FSM_STATE_IDLE,
						FSM_STATE_ERROR },
				/* LEARNING */{ FSM_STATE_LEARNING, FSM_STATE_IDLE,
						FSM_STATE_ERROR },
				/* ROUTING */{ FSM_STATE_ROUTING, FSM_STATE_IDLE,
						FSM_STATE_ERROR },
				/* KEY_ROTATION */{ FSM_STATE_KEY_ROTATION, FSM_STATE_IDLE,
						FSM_STATE_ERROR },
				/* ERROR */{ FSM_STATE_ERROR, FSM_STATE_IDLE, FSM_STATE_ERROR },
				/* SHUTDOWN */{ FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN,
						FSM_STATE_SHUTDOWN } };

static void fsm_init(FSMContext *fsm) {
	fsm->current = FSM_STATE_BOOT;
	fsm->previous = FSM_STATE_BOOT;
	fsm->entry_time = (unsigned long) time(NULL);
	fsm->last_event = FSM_EVENT_NONE;
	pthread_mutex_init(&fsm->fsm_mutex, NULL);
	pthread_cond_init(&fsm->fsm_cond, NULL);
}

static void fsm_process(FSMContext *fsm, FSMEvent event, unsigned long time) {
	FSMState new_state;

	pthread_mutex_lock(&fsm->fsm_mutex);

	if (event < FSM_EVENTS) {
		new_state = fsm_table[fsm->current][event];
		if (new_state != fsm->current) {
			fsm->previous = fsm->current;
			fsm->current = new_state;
			fsm->entry_time = time;
			fsm->last_event = event;
		}
	}

	pthread_cond_broadcast(&fsm->fsm_cond);
	pthread_mutex_unlock(&fsm->fsm_mutex);
}

/*=============================================================================
 * BOOT SEQUENCE
 *============================================================================*/

static void boot_init(BootSequence *boot) {
	boot->step = 0;
	boot->completed = 0;
	boot->errors = 0;
	boot->boot_complete = 0;
	boot->boot_successful = 0;
	boot->start_time = (unsigned long) time(NULL);
	memset(boot->timings, 0, sizeof(boot->timings));
	memset(boot->error_msg, 0, sizeof(boot->error_msg));
}

static int boot_step_cpu_check(EVOXCoreSystem *system) {
	(void) system;
#if defined(__AVX2__) && defined(__FMA__)
	printf("[BOOT] CPU: AVX2+FMA detected\n");
	return 1;
#else
    printf("[BOOT] ERROR: AVX2/FMA required\n");
    return 0;
#endif
}

static int boot_step_memory_check(EVOXCoreSystem *system) {
	struct sysinfo si;
	if (sysinfo(&si) == 0) {
		printf("[BOOT] Memory: %lu MB free\n",
				(unsigned long) (si.freeram >> 20));
		return (si.freeram > 256 * 1024 * 1024) ? 1 : 0;
	}
	return 1;
}

static int boot_step_security_init(EVOXCoreSystem *system) {
	system->crypto = security_init();
	return (system->crypto != NULL) ? 1 : 0;
}

static int boot_step_network_init(EVOXCoreSystem *system) {
	/* Initialize network if needed */
	(void) system;
	return 1;
}

static int boot_step_rendering_init(EVOXCoreSystem *system) {
	system->gl = rendering_init(1280, 720);
	if (system->gl) {
		system->rendering_enabled = 1;
		return 1;
	}
	return 0;
}

static int boot_step_audio_init(EVOXCoreSystem *system) {
	return audio_init(system);
}

static int boot_step_model_scan(EVOXCoreSystem *system) {
	return model_scan_and_load(system);
}

static int boot_step(BootSequence *boot, EVOXCoreSystem *system) {
	int ret = 0;
	unsigned long start = (unsigned long) time(NULL);

	printf("[BOOT] Step %u: ", boot->step);

	switch (boot->step) {
	case 0:
		printf("CPU Check\n");
		ret = boot_step_cpu_check(system);
		break;
	case 1:
		printf("Memory Check\n");
		ret = boot_step_memory_check(system);
		break;
	case 2:
		printf("Security Init\n");
		ret = boot_step_security_init(system);
		break;
	case 3:
		printf("Network Init\n");
		ret = boot_step_network_init(system);
		break;
	case 4:
		printf("Rendering Init\n");
		ret = boot_step_rendering_init(system);
		break;
	case 5:
		printf("Audio Init\n");
		ret = boot_step_audio_init(system);
		break;
	case 6:
		printf("Model Scan\n");
		ret = boot_step_model_scan(system);
		break;
	case 7:
		printf("Boot Complete\n");
		boot->boot_complete = 1;
		boot->boot_successful = (boot->errors == 0);
		ret = 1;
		break;
	default:
		ret = 0;
		break;
	}

	if (!ret)
		boot->errors++;

	boot->timings[boot->step] = difftime(time(NULL), start);
	printf("       → %.2fs\n", boot->timings[boot->step]);

	if (ret && boot->step < FSM_BOOT_STEPS - 1)
		boot->step++;

	return ret;
}

static void boot_execute(EVOXCoreSystem *system) {
	boot_init(&system->boot);
	while (!system->boot.boot_complete) {
		boot_step(&system->boot, system);
	}
}

/*=============================================================================
 * MAIN LOOP
 *============================================================================*/

static void main_loop(EVOXCoreSystem *system) {
	unsigned long last_key_check = 0;
	double last_update = 0.0;
	double update_interval = 1.0 / UPDATE_RATE_HZ;
	unsigned long now;
	double current_time;

	gettimeofday(&system->perf.start_time, NULL);

	while (!system->shutdown) {
		now = (unsigned long) time(NULL);
		current_time = get_time();

		/* Process FSM events */
		switch (system->fsm.current) {
		case FSM_STATE_BOOT:
			if (system->boot.boot_complete) {
				if (system->boot.boot_successful) {
					fsm_process(&system->fsm, FSM_EVENT_BOOT_COMPLETE, now);
				} else {
					fsm_process(&system->fsm, FSM_EVENT_BOOT_FAILED, now);
				}
			}
			break;

		case FSM_STATE_MODEL_LOAD:
			if (!system->model_loaded) {
				model_scan_and_load(system);
			}
			if (system->model_loaded) {
				fsm_process(&system->fsm, FSM_EVENT_MODEL_LOADED, now);
			}
			break;

		case FSM_STATE_IDLE:
			/* Check key rotation */
			if (now - last_key_check > 3600) {
				if (system->crypto
						&& security_check_expiry(system->crypto, now)) {
					fsm_process(&system->fsm, FSM_EVENT_KEY_EXPIRING, now);
				}
				last_key_check = now;
			}

			/* Trigger processing periodically */
			if (system->model_loaded
					&& (current_time - last_update) >= update_interval) {
				fsm_process(&system->fsm, FSM_EVENT_INFERENCE_REQUEST, now);
			}
			break;

		case FSM_STATE_PROCESSING:
			if (system->model_loaded && system->network) {
				neural_update_real_time(system);
				system->perf.total_inferences++;
			}
			fsm_process(&system->fsm, FSM_EVENT_BOOT_COMPLETE, now);
			last_update = current_time;
			break;

		case FSM_STATE_KEY_ROTATION:
			if (system->crypto) {
				security_rotate_keys(system->crypto);
				printf("[SECURITY] Keys rotated (%u)\n",
						system->crypto->rotations);
			}
			fsm_process(&system->fsm, FSM_EVENT_KEY_ROTATED, now);
			break;

		case FSM_STATE_SHUTDOWN:
			system->shutdown = 1;
			break;

		default:
			/* Auto-progress through init states */
			if (system->fsm.current < FSM_STATE_IDLE) {
				fsm_process(&system->fsm, FSM_EVENT_BOOT_COMPLETE, now);
			}
			break;
		}

		/* Render */
		if (system->gl && system->gl->is_allocated) {
			rendering_render(system);
		}

		/* Handle SDL events */
		if (system->gl && system->gl->window) {
			SDL_Event e;
			while (SDL_PollEvent(&e)) {
				if (e.type == SDL_QUIT) {
					fsm_process(&system->fsm, FSM_EVENT_SHUTDOWN_REQUEST, now);
				} else if (e.type == SDL_KEYDOWN) {
					if (e.key.keysym.sym == SDLK_ESCAPE) {
						fsm_process(&system->fsm, FSM_EVENT_SHUTDOWN_REQUEST,
								now);
					}
				}
			}
		}

		usleep(1000);
	}
}

/*=============================================================================
 * SYSTEM CLEANUP
 *============================================================================*/

static void system_cleanup(EVOXCoreSystem *system) {
	if (!system)
		return;

	printf("\n[CLEANUP] Shutting down...\n");

	if (system->network) {
		NeuralNetwork *net = system->network;
		free(net->nodes);
		free(net->synapses);
		free(net->node_activations);
		free(net->node_deltas);
		free(net->layer_outputs);
		free(net->attention_weights);
		free(net->expert_routing);
		free(net->expert_gates);
		free(net->activity_history);
		free(net->power_spectrum);
		pthread_spin_destroy(&net->network_lock);
		pthread_rwlock_destroy(&net->monitor_lock);
		free(net);
		printf("[CLEANUP] Neural network freed\n");
	}

	if (system->crypto) {
		EVP_CIPHER_CTX_free(system->crypto->cipher_ctx);
		EVP_MD_CTX_free(system->crypto->md_ctx);
		if (system->crypto->pkey)
			EVP_PKEY_free(system->crypto->pkey);
		pthread_mutex_destroy(&system->crypto->crypto_mutex);
		free(system->crypto);
		printf("[CLEANUP] Security context freed\n");
	}

	if (system->gl) {
		if (system->gl->gl_context)
			SDL_GL_DeleteContext(system->gl->gl_context);
		if (system->gl->window)
			SDL_DestroyWindow(system->gl->window);
		pthread_mutex_destroy(&system->gl->render_mutex);
		free(system->gl);
		printf("[CLEANUP] OpenGL context freed\n");
	}

	if (system->audio_context) {
		alcDestroyContext(system->audio_context);
	}
	if (system->audio_device) {
		alcCloseDevice(system->audio_device);
	}

	pthread_mutex_destroy(&system->fsm.fsm_mutex);
	pthread_cond_destroy(&system->fsm.fsm_cond);
	pthread_mutex_destroy(&system->system_mutex);
	pthread_rwlock_destroy(&system->model_lock);
	pthread_spin_destroy(&system->perf_lock);

	SDL_Quit();

	printf("[CLEANUP] Complete. Total inferences: %lu\n",
			system->perf.total_inferences);
	free(system);
}

/*=============================================================================
 * SYSTEM INITIALIZATION
 *============================================================================*/

static EVOXCoreSystem* system_init(void) {
	EVOXCoreSystem *system = (EVOXCoreSystem*) calloc(1,
			sizeof(EVOXCoreSystem));
	if (!system)
		return NULL;

	system->version_major = EVOX_VERSION_MAJOR;
	system->version_minor = EVOX_VERSION_MINOR;
	system->version_patch = EVOX_VERSION_PATCH;

	fsm_init(&system->fsm);
	five_axes_init(system);

	pthread_mutex_init(&system->system_mutex, NULL);
	pthread_rwlock_init(&system->model_lock, NULL);
	pthread_spin_init(&system->perf_lock, PTHREAD_PROCESS_PRIVATE);

	system->start_time = (unsigned long) time(NULL);

	printf(
			"\n╔══════════════════════════════════════════════════════════════╗\n");
	printf("║        EVOX AI CORE v%u.%u.%u - 5-AXES NEURAL SYSTEM         ║\n",
			system->version_major, system->version_minor,
			system->version_patch);
	printf(
			"║     Academic AI: MoE | R1 | V2 | Coder                       ║\n");
	printf(
			"║     Military-Grade Security | 28h Key Rotation               ║\n");
	printf(
			"║     Real-time Neural Monitoring | AVX-256 | FMA              ║\n");
	printf(
			"╚══════════════════════════════════════════════════════════════╝\n");
	printf("\n");

	return system;
}

/*=============================================================================
 * ENTRY POINT
 *============================================================================*/

int main(int argc, char *argv[]) {
	EVOXCoreSystem *system;

	srand((unsigned int) time(NULL));

	/* Initialize GLUT */
	glutInit(&argc, argv);

	/* Create models directory */
	mkdir(MODELS_DIRECTORY, 0755);

	/* Initialize system */
	system = system_init();
	if (!system) {
		fprintf(stderr, "FATAL: System initialization failed\n");
		return EXIT_FAILURE;
	}

	/* Execute boot sequence */
	boot_execute(system);

	/* Start main loop */
	printf("\n[SYSTEM] Entering main loop\n");
	printf("         Model directory: %s\n", MODELS_DIRECTORY);
	printf("         Press ESC to exit\n\n");

	main_loop(system);

	/* Cleanup */
	system_cleanup(system);

	printf("\n[SYSTEM] Terminated normally\n");
	return EXIT_SUCCESS;
}
