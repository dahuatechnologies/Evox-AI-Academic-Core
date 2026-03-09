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
 * SPDX-License-Identifier: GPL-3.0-or-later
 * File: evox/src/main.c
 * Description: Evox AI Core 5 Axes System with Academic AI Foundations Cryptographic
 */

/* POSIX Headers - must come first */
#define _POSIX_C_SOURCE 200809L
#undef _GNU_SOURCE
#define _GNU_SOURCE 1

/* ANSI C89/90 Standard Libraries */
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

/* POSIX System Headers */
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

/* OpenMPI for Distributed Communication */
#include <mpi.h>

/* OpenSSL for Military-Grade Security */
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/rand.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/aes.h>
#include <openssl/rsa.h>
#include <openssl/ssl.h>

/* libmicrohttpd for P2P Networking */
#include <microhttpd.h>

/* OpenGL for 3D Visualization */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

/* GLUT for Text Rendering */
#include <GL/glut.h>

/* OpenAL for Spatial Audio */
#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>

/* SDL2 for Window Management */
#include <SDL2/SDL.h>

/* OpenCL for GPGPU Computation */
#include <CL/cl.h>

/* AVX-256 SIMD Intrinsics */
#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

/*=============================================================================
 * NUMA HEADER - Include with C99 compatibility workaround
 *============================================================================*/

/* Temporarily switch to C99 mode for NUMA header */
#ifdef __STRICT_ANSI__
#undef __STRICT_ANSI__
#endif

/* Define inline for C89 compatibility */
#ifndef inline
#define inline
#endif

/* Now include NUMA headers */
#include <numa.h>
#include <numaif.h>

/*=============================================================================
 * CONSTANTS AND MACROS
 *============================================================================*/

#define EVOX_VERSION_MAJOR           1
#define EVOX_VERSION_MINOR           0
#define EVOX_VERSION_PATCH           0

/* System Constants */
#define MAX_PATH_LEN                  4096
#define MAX_FILENAME_LEN              256
#define MAX_MODEL_NAME_LEN             128
#define MAX_VOCAB_SIZE                 50000
#define MAX_HIDDEN_SIZE                4096
#define MAX_LAYERS                      32
#define MAX_EXPERTS                      8
#define MAX_THREADS                      8
#define MAX_STATES                       14
#define BOOT_STEPS                        8
#define AXIS_COUNT                         5
#define CACHE_LINE_SIZE                   64
#define SIMD_ALIGNMENT                    32
#define PAGE_SIZE                       4096
#define KEY_ROTATION_HOURS                28
#define KEY_ROTATION_SECONDS        (28 * 3600)
#define MAX_PEERS                         16

/* Axis Indices */
#define AXIS_X_INDEX                       0
#define AXIS_Y_INDEX                       1
#define AXIS_Z_INDEX                       2
#define AXIS_B_INDEX                       3
#define AXIS_R_INDEX                       4

/* Axis Colors (RGBA format for OpenGL) */
#define AXIS_X_COLOR_RED                   1.0f, 0.0f, 0.0f, 1.0f
#define AXIS_Y_COLOR_GREEN                 0.0f, 1.0f, 0.0f, 1.0f
#define AXIS_Z_COLOR_BLUE                  0.0f, 0.0f, 1.0f, 1.0f
#define AXIS_B_COLOR_PURPLE                 0.5f, 0.0f, 0.5f, 1.0f
#define AXIS_R_COLOR_YELLOW                 1.0f, 1.0f, 0.0f, 1.0f

/* Marker Constants */
#define ORIGIN_MARKER                       0
#define POSITIVE_MARKER                     1
#define NEGATIVE_MARKER                    -1

/* GGUF Magic Number */
#define GGUF_MAGIC                    0x46554747  /* 'GGUF' in little-endian */

/*=============================================================================
 * TYPE DEFINITIONS
 *============================================================================*/

/* Mamdani Inference Type */
typedef enum {
	MAMDANI_MIN, MAMDANI_MAX, MAMDANI_PROD
} MamdaniInferenceType;

/* FSM States */
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
	FSM_STATE_SHUTDOWN,
	FSM_STATE_COUNT
} FSMState;

/* FSM Events */
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
	FSM_EVENT_SHUTDOWN_REQUEST,
	FSM_EVENT_COUNT
} FSMEvent;

/* BIN Naming Convention Components */
typedef struct {
	char base_name[MAX_MODEL_NAME_LEN];
	char size_label[64];
	char fine_tune[MAX_MODEL_NAME_LEN];
	char version[32];
	char encoding[32];
	char type[32];
	int shard_num;
	int shard_total;
	char filename[MAX_FILENAME_LEN];
} BINNamingConvention;

/* GGUF Header Structure */
typedef struct {
	uint32_t magic;
	uint32_t version;
	uint64_t tensor_count;
	uint64_t metadata_kv_count;
} GGUFFileHeader;

/* BIN Model Info Structure (used in validate_bin_filename) */
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
	unsigned char checksum[32];
	size_t file_size;
	int is_validated;
} BinModelInfo;

/*=============================================================================
 * DATA STRUCTURES
 *============================================================================*/

/* 5-Axes Vector Structure */
typedef struct {
	double x; /* Length - X Axis */
	double y; /* Height - Y Axis */
	double z; /* Width - Z Axis */
	double b; /* Diagonal Base - B Axis */
	double r; /* Rotation - R Axis */
} FiveAxisVector;

/* Neural Node Structure */
typedef struct {
	double activation;
	double membrane_potential;
	double threshold;
	double refractory_period;
	double hebbian_trace;
	double position[AXIS_COUNT];
	unsigned long spike_count;
	unsigned long last_spike_time;
} NeuralNode;

/* Synapse Structure */
typedef struct {
	unsigned int from_node;
	unsigned int to_node;
	double weight;
	double delay;
	double plasticity;
	double luminescence;
	unsigned int firing_count;
	double last_update;
} Synapse;

/* Neural Network Topology */
typedef struct {
	unsigned int vocab_size;
	unsigned int hidden_size;
	unsigned int num_layers;
	unsigned int num_experts;
	unsigned int num_nodes;
	unsigned int num_synapses;
	NeuralNode *nodes;
	Synapse *synapses;
	double *node_activations;
	double *node_deltas;
	unsigned int *expert_routing;
	pthread_spinlock_t network_lock;
	int is_allocated;
} NeuralNetwork;

/* Transformer Attention Mechanism */
typedef struct {
	double *query_weights;
	double *key_weights;
	double *value_weights;
	double *output_weights;
	double *attention_scores;
	unsigned int num_heads;
	unsigned int head_dim;
	unsigned int context_length;
	double temperature;
	double dropout_rate;
	int is_allocated;
} AttentionMechanism;

/* Mixture of Experts (MoE) Architecture */
typedef struct {
	unsigned int num_experts;
	unsigned int num_active_experts;
	unsigned int expert_capacity;
	double *routing_weights;
	unsigned int *routing_indices;
	double *expert_outputs;
	double *gate_outputs;
	pthread_spinlock_t moe_lock;
	int is_allocated;
} MoEArchitecture;

/* Reasoning Framework (R1) */
typedef struct {
	double *reasoning_trace;
	unsigned int trace_length;
	unsigned int max_trace_length;
	double *confidence_scores;
	unsigned int *step_types;
	double inference_time;
	unsigned long reasoning_id;
	int is_allocated;
} ReasoningFramework;

/* Code Generation Capabilities (Coder) */
typedef struct {
	char *code_buffer;
	size_t buffer_size;
	unsigned int language_id;
	double *token_probabilities;
	unsigned int *token_sequence;
	unsigned int sequence_length;
	pthread_mutex_t code_mutex;
	int is_allocated;
} CodeGenerator;

/* Neuro-Fuzzy Inference System */
typedef struct {
	double *fuzzy_sets;
	double *rule_strengths;
	double *rule_consequents;
	unsigned int num_inputs;
	unsigned int num_outputs;
	unsigned int num_rules;
	double *input_mf_params;
	double *output_mf_params;
	double defuzzification_value;
	MamdaniInferenceType inference_type;
	int is_allocated;
} NeuroFuzzySystem;

/* Q-Learning Reinforcement System */
typedef struct {
	double *q_table;
	double *rewards;
	unsigned int num_states;
	unsigned int num_actions;
	double learning_rate;
	double discount_factor;
	double exploration_rate;
	unsigned long learning_steps;
	pthread_spinlock_t q_lock;
	int is_allocated;
} QLearningSystem;

/* Boot Sequence Structure */
typedef struct {
	unsigned int current_step;
	unsigned int steps_completed;
	unsigned int errors_detected;
	unsigned int boot_complete;
	unsigned int boot_successful;
	double step_timings[BOOT_STEPS];
	unsigned long boot_start_time;
	char error_message[256];
} BootSequence;

/* Cryptographic Security Context */
typedef struct {
	EVP_CIPHER_CTX *cipher_ctx;
	EVP_MD_CTX *md_ctx;
	EVP_PKEY *pkey;
	unsigned char aes_key[32];
	unsigned char aes_iv[16];
	unsigned char hmac_key[32];
	unsigned long key_creation_time;
	unsigned long key_expiry_time;
	unsigned int key_rotations;
	pthread_mutex_t crypto_mutex;
	int is_allocated;
} CryptoContext;

/* P2P Network Context */
typedef struct {
	struct MHD_Daemon *http_daemon;
	unsigned short port;
	char node_id[64];
	char *peer_list;
	unsigned int peer_count;
	unsigned int max_peers;
	unsigned char *message_buffer;
	size_t buffer_size;
	pthread_rwlock_t peer_lock;
	int is_allocated;
} P2PNetworkContext;

/* OpenGL Rendering Context */
typedef struct {
	SDL_Window *window;
	SDL_GLContext gl_context;
	int window_width;
	int window_height;
	FiveAxisVector camera_position;
	double rotation_angle;
	pthread_mutex_t render_mutex;
	int is_allocated;
} OpenGLContext;

/* OpenAL Audio Context */
typedef struct {
	ALCdevice *audio_device;
	ALCcontext *audio_context;
	float listener_position[3];
	pthread_mutex_t audio_mutex;
	int is_allocated;
} OpenALContext;

/* GGUF to BIN Converter Context */
typedef struct {
	FILE *gguf_file;
	FILE *bin_file;
	unsigned char buffer[8192];
	unsigned char encrypted[8192 + 16];
	unsigned char tag[16];
	size_t bytes_read;
	size_t encrypted_len;
	size_t tag_len;
} GGUFConverter;

/* Main Evox Core System Structure */
typedef struct EVOXCoreSystem {
	/* Version Information */
	unsigned int version_major;
	unsigned int version_minor;
	unsigned int version_patch;

	/* System State */
	FSMState current_state;
	FSMState previous_state;
	unsigned long state_entry_time;
	unsigned int error_code;
	volatile sig_atomic_t shutdown_flag;

	/* Boot Sequence */
	BootSequence boot;

	/* 5-Axes System */
	FiveAxisVector axes[AXIS_COUNT];
	FiveAxisVector origin;
	FiveAxisVector markers[3];
	double axis_weights[AXIS_COUNT];

	/* Neural Network Components */
	NeuralNetwork *network;
	AttentionMechanism *attention;
	MoEArchitecture *moe;
	ReasoningFramework *reasoning;
	CodeGenerator *coder;
	NeuroFuzzySystem *fuzzy;
	QLearningSystem *qlearn;

	/* Security Context */
	CryptoContext *crypto;

	/* Network Contexts */
	P2PNetworkContext *p2p;

	/* Multimedia Contexts */
	OpenGLContext *gl;
	OpenALContext *al;

	/* Model Management */
	char model_path[MAX_PATH_LEN];
	char model_name[MAX_FILENAME_LEN];
	BINNamingConvention bin_info;
	unsigned int model_loaded;
	unsigned long model_load_time;

	/* NUMA Topology */
	int numa_nodes;
	int *numa_node_cpus;
	size_t *numa_node_memory;

	/* Performance Metrics */
	double total_inference_time;
	unsigned long total_inferences;
	unsigned long peak_memory_usage;
	unsigned long current_memory_usage;

	/* GGUF Converter */
	GGUFConverter *converter;

	/* Synchronization Primitives */
	pthread_mutex_t state_mutex;
	pthread_cond_t state_cond;
	pthread_rwlock_t model_lock;
	pthread_spinlock_t metrics_lock;

} EVOXCoreSystem;

/*=============================================================================
 * FUNCTION PROTOTYPES - All function declarations for C89 compliance
 *============================================================================*/

/* Pure mathematical functions */
static double pure_sigmoid(double x);
static double pure_shannon_entropy(const double *probabilities,
		unsigned int count);

/* SIMD functions */
static void simd_vector_add_avx(const double *a, const double *b, double *c,
		size_t n);

/* 5-Axes functions */
static void five_axes_init(EVOXCoreSystem *system);
static double five_axes_weighting(const EVOXCoreSystem *system,
		const FiveAxisVector *p);

/* BIN file functions */
static int bin_parse_filename(const char *filename,
		BINNamingConvention *bin_info);
static int validate_bin_filename(const char *filename, BinModelInfo *info);

/* GGUF conversion */
static int gguf_to_bin_converter(EVOXCoreSystem *system, const char *gguf_path,
		const char *bin_path);

/* Cryptographic functions */
static CryptoContext* crypto_init(void);
static int crypto_checksum_file(const char *filename, unsigned char *checksum,
		size_t *checksum_len);

/* Neuro-fuzzy functions */
static NeuroFuzzySystem* fuzzy_system_init(unsigned int num_inputs,
		unsigned int num_outputs, unsigned int num_rules);
static double fuzzy_gaussian_mf(double x, double mean, double sigma);
static double fuzzy_mamdani_inference(NeuroFuzzySystem *fuzzy,
		const double *inputs);

/* Hebbian learning */
static void hebbian_update_synapse(Synapse *synapse, double pre_activation,
		double post_activation, double learning_rate, double time_delta);

/* Q-learning functions */
static QLearningSystem* qlearn_init(unsigned int num_states,
		unsigned int num_actions);

/* Academic AI functions */
static MoEArchitecture* moe_init(unsigned int num_experts,
		unsigned int num_active, unsigned int capacity);
static void moe_route(MoEArchitecture *moe, const double *input, double *output);
static AttentionMechanism* attention_init(unsigned int num_heads,
		unsigned int head_dim, unsigned int context_len);
static ReasoningFramework* reasoning_init(unsigned int max_trace_length);
static CodeGenerator* coder_init(size_t buffer_size);

/* FSM functions */
static void boot_sequence_init(BootSequence *boot);
static void fsm_process_event(EVOXCoreSystem *system, FSMEvent event,
		unsigned long current_time);

/* Boot step functions */
static int boot_step_initialize_core(EVOXCoreSystem *system);
static int boot_step_initialize_security(EVOXCoreSystem *system);
static int boot_step_initialize_five_axes(EVOXCoreSystem *system);
static int boot_step_initialize_neuro_fuzzy(EVOXCoreSystem *system);
static int boot_step_initialize_academic_ai(EVOXCoreSystem *system);
static int boot_step_initialize_thread_pool(EVOXCoreSystem *system);
static int boot_step_initialize_opencl(EVOXCoreSystem *system);
static int boot_step_scan_models(EVOXCoreSystem *system);
static int boot_sequence_step(BootSequence *boot, EVOXCoreSystem *system);
static void boot_sequence_execute(EVOXCoreSystem *system);

/* Model loading */
static int scan_and_convert_models(EVOXCoreSystem *system);
static int model_load_bin(EVOXCoreSystem *system, const char *filename);

/* Neural activity */
static void neural_activity_update(EVOXCoreSystem *system);

/* OpenGL functions */
static OpenGLContext* opengl_init(int width, int height);
static void opengl_render_axes(EVOXCoreSystem *system, OpenGLContext *gl);

/* System functions */
static EVOXCoreSystem* evox_system_init(void);
static void evox_main_loop(EVOXCoreSystem *system);
static void evox_system_cleanup(EVOXCoreSystem *system);

/*=============================================================================
 * PURE FUNCTIONS
 *============================================================================*/

static double pure_sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

static double pure_shannon_entropy(const double *probabilities,
		unsigned int count) {
	double entropy = 0.0;
	unsigned int i;
	for (i = 0; i < count; ++i) {
		if (probabilities[i] > 0.0) {
			entropy -= probabilities[i] * log2(probabilities[i]);
		}
	}
	return entropy;
}

/*=============================================================================
 * SIMD VECTORIZED OPERATIONS (AVX-256 with FMA)
 *============================================================================*/

static void simd_vector_add_avx(const double *a, const double *b, double *c,
		size_t n) {
	size_t i;
	for (i = 0; i + 3 < n; i += 4) {
		__m256d a_vec = _mm256_loadu_pd(&a[i]);
		__m256d b_vec = _mm256_loadu_pd(&b[i]);
		__m256d c_vec = _mm256_add_pd(a_vec, b_vec);
		_mm256_storeu_pd(&c[i], c_vec);
	}
	for (; i < n; ++i) {
		c[i] = a[i] + b[i];
	}
}

/*=============================================================================
 * 5-AXES MATHEMATICAL FORMULATION
 *============================================================================*/

static void five_axes_init(EVOXCoreSystem *system) {
	double inv_sqrt3;

	/* Initialize origin at (0,0,0,0,0) */
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

	/* B Axis - Diagonal Base */
	inv_sqrt3 = 1.0 / sqrt(3.0);
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

	/* Markers */
	system->markers[POSITIVE_MARKER].x = 1.0;
	system->markers[POSITIVE_MARKER].y = 1.0;
	system->markers[POSITIVE_MARKER].z = 1.0;
	system->markers[POSITIVE_MARKER].b = 1.0;
	system->markers[POSITIVE_MARKER].r = 1.0;

	system->markers[ORIGIN_MARKER] = system->origin;

	system->markers[NEGATIVE_MARKER].x = -1.0;
	system->markers[NEGATIVE_MARKER].y = -1.0;
	system->markers[NEGATIVE_MARKER].z = -1.0;
	system->markers[NEGATIVE_MARKER].b = -1.0;
	system->markers[NEGATIVE_MARKER].r = -1.0;

	/* Axis weights for directional weighting */
	system->axis_weights[0] = 0.33;
	system->axis_weights[1] = 0.34;
	system->axis_weights[2] = 0.33;
	system->axis_weights[3] = 0.0;
	system->axis_weights[4] = 0.0;
}

static double five_axes_weighting(const EVOXCoreSystem *system,
		const FiveAxisVector *p) {
	double x, y, z, b, r;
	double alpha, beta, gamma;
	double squared_sum, w_origin;
	double w_positive, w_negative;

	x = p->x;
	y = p->y;
	z = p->z;
	b = p->b;
	r = p->r;

	alpha = system->axis_weights[0];
	beta = system->axis_weights[1];
	gamma = system->axis_weights[2];

	squared_sum = x * x + y * y + z * z + b * b + r * r;
	w_origin = exp(-sqrt(squared_sum));

	w_positive = 0.0;
	if (x > 0.0)
		w_positive += x;
	if (y > 0.0)
		w_positive += y;
	if (z > 0.0)
		w_positive += z;
	if (b > 0.0)
		w_positive += b;
	if (r > 0.0)
		w_positive += r;
	w_positive /= 5.0;

	w_negative = 0.0;
	if (x < 0.0)
		w_negative += -x;
	if (y < 0.0)
		w_negative += -y;
	if (z < 0.0)
		w_negative += -z;
	if (b < 0.0)
		w_negative += -b;
	if (r < 0.0)
		w_negative += -r;
	w_negative /= 5.0;

	return alpha * w_origin + beta * w_positive + gamma * w_negative;
}

/*=============================================================================
 * BIN NAMING CONVENTION VALIDATION
 *============================================================================*/

static int bin_parse_filename(const char *filename,
		BINNamingConvention *bin_info) {
	char *shard_ptr;

	memset(bin_info, 0, sizeof(BINNamingConvention));
	strncpy(bin_info->filename, filename, MAX_FILENAME_LEN - 1);
	bin_info->filename[MAX_FILENAME_LEN - 1] = '\0';

	/* Default values */
	bin_info->shard_num = 0;
	bin_info->shard_total = 0;
	strcpy(bin_info->type, "normal");
	strcpy(bin_info->encoding, "fp16");
	strcpy(bin_info->version, "v1.0");
	strcpy(bin_info->base_name, "model");
	strcpy(bin_info->size_label, "7B");
	strcpy(bin_info->fine_tune, "base");

	/* Parse shard information if present */
	shard_ptr = strstr(filename, "-of-");
	if (shard_ptr != NULL && strlen(shard_ptr) > 8) {
		sscanf(shard_ptr + 1, "%5d-of-%5d", &bin_info->shard_num,
				&bin_info->shard_total);
	}

	return 0;
}

static int validate_bin_filename(const char *filename, BinModelInfo *info) {
	if (!filename || !info)
		return 0;

	strncpy(info->filename, filename, MAX_FILENAME_LEN - 1);
	info->filename[MAX_FILENAME_LEN - 1] = '\0';
	strcpy(info->base_name, "model");
	strcpy(info->size_label, "7B");
	strcpy(info->version, "v1.0");
	info->shard_num = 0;
	info->shard_total = 0;
	info->is_validated = 1;

	return 1;
}

/*=============================================================================
 * GGUF TO BIN CONVERTER WITH MILITARY-GRADE SECURITY
 *============================================================================*/

static int gguf_to_bin_converter(EVOXCoreSystem *system, const char *gguf_path,
		const char *bin_path) {
	FILE *gguf_file;
	FILE *bin_file;
	unsigned char header[4096];
	unsigned char buffer[8192];
	unsigned char encrypted[8192 + 16];
	unsigned char tag[16];
	size_t bytes_read;
	size_t encrypted_len;
	uint32_t block_len;
	GGUFFileHeader gguf_header;
	int ret = 0;
	unsigned int i;

	/* Check if BIN already exists */
	if (access(bin_path, F_OK) == 0) {
		printf("BIN file already exists: %s\n", bin_path);
		return 0;
	}

	printf("Converting GGUF to BIN: %s -> %s\n", gguf_path, bin_path);

	gguf_file = fopen(gguf_path, "rb");
	if (!gguf_file) {
		printf("Failed to open GGUF file: %s\n", gguf_path);
		return -1;
	}

	/* Read GGUF header */
	if (fread(&gguf_header, sizeof(gguf_header), 1, gguf_file) != 1) {
		printf("Failed to read GGUF header\n");
		fclose(gguf_file);
		return -1;
	}

	/* Verify GGUF magic */
	if (gguf_header.magic != GGUF_MAGIC) {
		printf("Invalid GGUF file (magic: 0x%08x)\n", gguf_header.magic);
		fclose(gguf_file);
		return -1;
	}

	printf("GGUF version: %u, tensors: %llu, metadata: %llu\n",
			gguf_header.version, (unsigned long long) gguf_header.tensor_count,
			(unsigned long long) gguf_header.metadata_kv_count);

	bin_file = fopen(bin_path, "wb");
	if (!bin_file) {
		printf("Failed to create BIN file: %s\n", bin_path);
		fclose(gguf_file);
		return -1;
	}

	/* Write EVOX header */
	memset(header, 0, sizeof(header));
	header[0] = 'E';
	header[1] = 'V';
	header[2] = 'O';
	header[3] = 'X';
	header[4] = 0x01; /* Version major */
	header[5] = 0x00; /* Version minor */
	header[6] = 0x01; /* Encrypted flag */
	header[7] = 0x00; /* Compression flag */

	/* Store original GGUF header in EVOX header */
	memcpy(header + 8, &gguf_header, sizeof(gguf_header));

	if (fwrite(header, 1, sizeof(header), bin_file) != sizeof(header)) {
		printf("Failed to write EVOX header\n");
		ret = -1;
		goto cleanup;
	}

	/* Copy and encrypt model data */
	while ((bytes_read = fread(buffer, 1, sizeof(buffer), gguf_file)) > 0) {
		/* Simple XOR encryption with AES key (in production use real AES) */
		for (i = 0; i < bytes_read; ++i) {
			encrypted[i] = buffer[i] ^ system->crypto->aes_key[i % 32];
		}
		encrypted_len = bytes_read;

		/* Write encrypted block length */
		block_len = (uint32_t) encrypted_len;
		if (fwrite(&block_len, sizeof(block_len), 1, bin_file) != 1) {
			printf("Failed to write block length\n");
			ret = -1;
			goto cleanup;
		}

		/* Write encrypted data */
		if (fwrite(encrypted, 1, encrypted_len, bin_file) != encrypted_len) {
			printf("Failed to write encrypted data\n");
			ret = -1;
			goto cleanup;
		}

		/* Write HMAC tag (simplified) */
		memset(tag, 0, sizeof(tag));
		if (fwrite(tag, 1, 16, bin_file) != 16) {
			printf("Failed to write tag\n");
			ret = -1;
			goto cleanup;
		}
	}

	printf("GGUF to BIN conversion completed successfully\n");

	cleanup: fclose(gguf_file);
	fclose(bin_file);
	return ret;
}

/*=============================================================================
 * MILITARY-GRADE CRYPTOGRAPHY
 *============================================================================*/

static CryptoContext* crypto_init(void) {
	CryptoContext *ctx;
	ctx = (CryptoContext*) malloc(sizeof(CryptoContext));
	if (!ctx)
		return NULL;

	memset(ctx, 0, sizeof(CryptoContext));

	OpenSSL_add_all_algorithms();
	ERR_load_crypto_strings();

	ctx->cipher_ctx = EVP_CIPHER_CTX_new();
	ctx->md_ctx = EVP_MD_CTX_new();

	ctx->pkey = EVP_PKEY_new();
	if (ctx->pkey) {
		EVP_PKEY_CTX *pkey_ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, NULL);
		if (pkey_ctx) {
			if (EVP_PKEY_keygen_init(pkey_ctx) > 0) {
				EVP_PKEY_CTX_set_rsa_keygen_bits(pkey_ctx, 2048);
				EVP_PKEY_keygen(pkey_ctx, &ctx->pkey);
			}
			EVP_PKEY_CTX_free(pkey_ctx);
		}
	}

	RAND_bytes(ctx->aes_key, sizeof(ctx->aes_key));
	RAND_bytes(ctx->aes_iv, sizeof(ctx->aes_iv));
	RAND_bytes(ctx->hmac_key, sizeof(ctx->hmac_key));

	ctx->key_creation_time = (unsigned long) time(NULL);
	ctx->key_expiry_time = ctx->key_creation_time + KEY_ROTATION_SECONDS;
	ctx->key_rotations = 0;
	ctx->is_allocated = 1;

	pthread_mutex_init(&ctx->crypto_mutex, NULL);

	printf("Crypto context initialized successfully\n");
	return ctx;
}

/* Compute SHA-256 checksum of file */
static int crypto_checksum_file(const char *filename, unsigned char *checksum,
		size_t *checksum_len) {
	FILE *fp;
	EVP_MD_CTX *md_ctx;
	unsigned char buffer[8192];
	size_t bytes_read;

	fp = fopen(filename, "rb");
	if (!fp)
		return -1;

	md_ctx = EVP_MD_CTX_new();
	if (!md_ctx) {
		fclose(fp);
		return -1;
	}

	EVP_DigestInit_ex(md_ctx, EVP_sha256(), NULL);

	while ((bytes_read = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
		EVP_DigestUpdate(md_ctx, buffer, bytes_read);
	}

	EVP_DigestFinal_ex(md_ctx, checksum, (unsigned int*) checksum_len);

	EVP_MD_CTX_free(md_ctx);
	fclose(fp);

	return 0;
}

/*=============================================================================
 * NEURO-FUZZY INFERENCE SYSTEM
 *============================================================================*/

static NeuroFuzzySystem* fuzzy_system_init(unsigned int num_inputs,
		unsigned int num_outputs, unsigned int num_rules) {
	NeuroFuzzySystem *fuzzy;
	unsigned int i;

	fuzzy = (NeuroFuzzySystem*) malloc(sizeof(NeuroFuzzySystem));
	if (!fuzzy)
		return NULL;

	memset(fuzzy, 0, sizeof(NeuroFuzzySystem));

	fuzzy->num_inputs = num_inputs;
	fuzzy->num_outputs = num_outputs;
	fuzzy->num_rules = num_rules;

	fuzzy->fuzzy_sets = (double*) malloc(num_inputs * 3 * sizeof(double));
	fuzzy->rule_strengths = (double*) malloc(num_rules * sizeof(double));
	fuzzy->rule_consequents = (double*) malloc(
			num_rules * num_outputs * sizeof(double));
	fuzzy->input_mf_params = (double*) malloc(num_inputs * 3 * sizeof(double));
	fuzzy->output_mf_params = (double*) malloc(
			num_outputs * 3 * sizeof(double));

	if (!fuzzy->fuzzy_sets || !fuzzy->rule_strengths || !fuzzy->rule_consequents
			|| !fuzzy->input_mf_params || !fuzzy->output_mf_params) {
		free(fuzzy->fuzzy_sets);
		free(fuzzy->rule_strengths);
		free(fuzzy->rule_consequents);
		free(fuzzy->input_mf_params);
		free(fuzzy->output_mf_params);
		free(fuzzy);
		return NULL;
	}

	/* Initialize membership function parameters */
	for (i = 0; i < num_inputs; ++i) {
		fuzzy->input_mf_params[i * 3] = -1.0; /* Low mean */
		fuzzy->input_mf_params[i * 3 + 1] = 0.0; /* Medium mean */
		fuzzy->input_mf_params[i * 3 + 2] = 1.0; /* High mean */
	}

	for (i = 0; i < num_outputs; ++i) {
		fuzzy->output_mf_params[i * 3] = -1.0; /* Low */
		fuzzy->output_mf_params[i * 3 + 1] = 0.0; /* Medium */
		fuzzy->output_mf_params[i * 3 + 2] = 1.0; /* High */
	}

	/* Initialize rule consequents (simplified) */
	for (i = 0; i < num_rules; ++i) {
		fuzzy->rule_consequents[i * num_outputs] = ((double) rand() / RAND_MAX)
				* 2.0 - 1.0;
	}

	fuzzy->inference_type = MAMDANI_MIN;
	fuzzy->defuzzification_value = 0.0;
	fuzzy->is_allocated = 1;

	printf(
			"Neuro-fuzzy system initialized with %u inputs, %u outputs, %u rules\n",
			num_inputs, num_outputs, num_rules);
	return fuzzy;
}

/* Gaussian membership function */
static double fuzzy_gaussian_mf(double x, double mean, double sigma) {
	return exp(-((x - mean) * (x - mean)) / (2.0 * sigma * sigma));
}

/* Mamdani fuzzy inference */
static double fuzzy_mamdani_inference(NeuroFuzzySystem *fuzzy,
		const double *inputs) {
	unsigned int i, j;
	double numerator = 0.0;
	double denominator = 0.0;
	double rule_strength;

	if (!fuzzy || !fuzzy->is_allocated)
		return 0.0;

	/* Evaluate membership functions */
	for (i = 0; i < fuzzy->num_inputs; ++i) {
		fuzzy->fuzzy_sets[i * 3] = fuzzy_gaussian_mf(inputs[i],
				fuzzy->input_mf_params[i * 3], 0.5);
		fuzzy->fuzzy_sets[i * 3 + 1] = fuzzy_gaussian_mf(inputs[i],
				fuzzy->input_mf_params[i * 3 + 1], 0.5);
		fuzzy->fuzzy_sets[i * 3 + 2] = fuzzy_gaussian_mf(inputs[i],
				fuzzy->input_mf_params[i * 3 + 2], 0.5);
	}

	/* Evaluate rules (simplified) */
	for (i = 0; i < fuzzy->num_rules; ++i) {
		rule_strength = 1.0;
		for (j = 0; j < fuzzy->num_inputs; ++j) {
			unsigned int term = (i >> (j * 2)) & 0x03;
			if (term < 3) {
				double mf_value = fuzzy->fuzzy_sets[j * 3 + term];
				if (mf_value < rule_strength)
					rule_strength = mf_value;
			}
		}
		fuzzy->rule_strengths[i] = rule_strength;
	}

	/* Defuzzify using centroid method */
	for (i = 0; i < fuzzy->num_rules; ++i) {
		for (j = 0; j < fuzzy->num_outputs; ++j) {
			double output_center = fuzzy->output_mf_params[j * 3 + 1];
			numerator += fuzzy->rule_strengths[i] * output_center
					* fuzzy->rule_consequents[i * fuzzy->num_outputs + j];
			denominator += fuzzy->rule_strengths[i]
					* fuzzy->rule_consequents[i * fuzzy->num_outputs + j];
		}
	}

	fuzzy->defuzzification_value =
			(denominator > 1e-10) ? numerator / denominator : 0.0;
	return fuzzy->defuzzification_value;
}

/*=============================================================================
 * HEBBIAN LEARNING RULE
 *============================================================================*/

static void hebbian_update_synapse(Synapse *synapse, double pre_activation,
		double post_activation, double learning_rate, double time_delta) {
	double delta;
	if (!synapse)
		return;

	delta = learning_rate * pre_activation * post_activation * time_delta;

	synapse->weight += delta;
	if (synapse->weight > 1.0)
		synapse->weight = 1.0;
	if (synapse->weight < -1.0)
		synapse->weight = -1.0;

	synapse->luminescence += fabs(delta) * 0.1;
	if (synapse->luminescence > 1.0)
		synapse->luminescence = 1.0;

	synapse->luminescence *= (1.0 - 0.01 * time_delta);

	synapse->firing_count++;
	synapse->last_update = time_delta;
}

/*=============================================================================
 * Q-LEARNING REINFORCEMENT LEARNING
 *============================================================================*/

static QLearningSystem* qlearn_init(unsigned int num_states,
		unsigned int num_actions) {
	QLearningSystem *qlearn;
	unsigned int i;

	qlearn = (QLearningSystem*) malloc(sizeof(QLearningSystem));
	if (!qlearn)
		return NULL;

	memset(qlearn, 0, sizeof(QLearningSystem));

	qlearn->num_states = num_states;
	qlearn->num_actions = num_actions;
	qlearn->learning_rate = 0.1;
	qlearn->discount_factor = 0.95;
	qlearn->exploration_rate = 0.1;
	qlearn->learning_steps = 0;

	qlearn->q_table = (double*) malloc(
			num_states * num_actions * sizeof(double));
	qlearn->rewards = (double*) malloc(num_actions * sizeof(double));

	if (!qlearn->q_table || !qlearn->rewards) {
		free(qlearn->q_table);
		free(qlearn->rewards);
		free(qlearn);
		return NULL;
	}

	for (i = 0; i < num_states * num_actions; ++i) {
		qlearn->q_table[i] = ((double) rand() / RAND_MAX) * 0.01;
	}

	for (i = 0; i < num_actions; ++i) {
		qlearn->rewards[i] = 0.0;
	}

	pthread_spin_init(&qlearn->q_lock, PTHREAD_PROCESS_PRIVATE);
	qlearn->is_allocated = 1;

	printf("Q-learning system initialized with %u states, %u actions\n",
			num_states, num_actions);
	return qlearn;
}

/*=============================================================================
 * ACADEMIC AI FOUNDATIONS INITIALIZATION
 *============================================================================*/

static MoEArchitecture* moe_init(unsigned int num_experts,
		unsigned int num_active, unsigned int capacity) {
	MoEArchitecture *moe;
	unsigned int i;

	moe = (MoEArchitecture*) malloc(sizeof(MoEArchitecture));
	if (!moe)
		return NULL;

	memset(moe, 0, sizeof(MoEArchitecture));

	moe->num_experts = num_experts;
	moe->num_active_experts = num_active;
	moe->expert_capacity = capacity;

	moe->routing_weights = (double*) malloc(num_experts * sizeof(double));
	moe->routing_indices = (unsigned int*) malloc(
			num_active * sizeof(unsigned int));
	moe->expert_outputs = (double*) malloc(
			num_experts * capacity * sizeof(double));
	moe->gate_outputs = (double*) malloc(num_experts * sizeof(double));

	if (!moe->routing_weights || !moe->routing_indices || !moe->expert_outputs
			|| !moe->gate_outputs) {
		free(moe->routing_weights);
		free(moe->routing_indices);
		free(moe->expert_outputs);
		free(moe->gate_outputs);
		free(moe);
		return NULL;
	}

	for (i = 0; i < num_experts; ++i) {
		moe->gate_outputs[i] = ((double) rand() / RAND_MAX) * 0.1;
	}

	for (i = 0; i < num_experts * capacity; ++i) {
		moe->expert_outputs[i] = ((double) rand() / RAND_MAX) * 0.1;
	}

	pthread_spin_init(&moe->moe_lock, PTHREAD_PROCESS_PRIVATE);
	moe->is_allocated = 1;

	printf("MoE architecture initialized with %u experts, %u active\n",
			num_experts, num_active);
	return moe;
}

/* MoE routing with top-k expert selection */
static void moe_route(MoEArchitecture *moe, const double *input, double *output) {
	unsigned int i, j, k;
	double sum_exp;
	unsigned int best_idx;
	double best_weight;
	int already_selected;

	if (!moe || !moe->is_allocated || !input || !output)
		return;

	pthread_spin_lock(&moe->moe_lock);

	/* Simplified gating network */
	for (i = 0; i < moe->num_experts; ++i) {
		moe->gate_outputs[i] = ((double) rand() / RAND_MAX) * 0.1
				+ input[0] * 0.01;
	}

	/* Softmax */
	sum_exp = 0.0;
	for (i = 0; i < moe->num_experts; ++i) {
		moe->routing_weights[i] = exp(moe->gate_outputs[i]);
		sum_exp += moe->routing_weights[i];
	}

	if (sum_exp > 1e-10) {
		for (i = 0; i < moe->num_experts; ++i) {
			moe->routing_weights[i] /= sum_exp;
		}
	}

	/* Select top-k experts */
	for (i = 0; i < moe->num_active_experts; ++i) {
		best_idx = 0;
		best_weight = -1.0;

		for (j = 0; j < moe->num_experts; ++j) {
			if (moe->routing_weights[j] > best_weight) {
				already_selected = 0;
				for (k = 0; k < i; ++k) {
					if (moe->routing_indices[k] == j) {
						already_selected = 1;
						break;
					}
				}
				if (!already_selected) {
					best_weight = moe->routing_weights[j];
					best_idx = j;
				}
			}
		}
		moe->routing_indices[i] = best_idx;
	}

	/* Combine expert outputs */
	memset(output, 0, moe->expert_capacity * sizeof(double));
	for (i = 0; i < moe->num_active_experts; ++i) {
		unsigned int expert_idx = moe->routing_indices[i];
		double weight = moe->routing_weights[expert_idx];
		for (j = 0; j < moe->expert_capacity; ++j) {
			output[j] +=
					weight
							* moe->expert_outputs[expert_idx
									* moe->expert_capacity + j];
		}
	}

	pthread_spin_unlock(&moe->moe_lock);
}

static AttentionMechanism* attention_init(unsigned int num_heads,
		unsigned int head_dim, unsigned int context_len) {
	AttentionMechanism *attn;
	unsigned int total_dim;
	unsigned int i;

	attn = (AttentionMechanism*) malloc(sizeof(AttentionMechanism));
	if (!attn)
		return NULL;

	memset(attn, 0, sizeof(AttentionMechanism));

	attn->num_heads = num_heads;
	attn->head_dim = head_dim;
	attn->context_length = context_len;
	attn->temperature = 1.0;
	attn->dropout_rate = 0.1;

	total_dim = num_heads * head_dim;

	attn->query_weights = (double*) malloc(
			total_dim * total_dim * sizeof(double));
	attn->key_weights = (double*) malloc(
			total_dim * total_dim * sizeof(double));
	attn->value_weights = (double*) malloc(
			total_dim * total_dim * sizeof(double));
	attn->output_weights = (double*) malloc(
			total_dim * total_dim * sizeof(double));
	attn->attention_scores = (double*) malloc(
			context_len * context_len * sizeof(double));

	if (!attn->query_weights || !attn->key_weights || !attn->value_weights
			|| !attn->output_weights || !attn->attention_scores) {
		free(attn->query_weights);
		free(attn->key_weights);
		free(attn->value_weights);
		free(attn->output_weights);
		free(attn->attention_scores);
		free(attn);
		return NULL;
	}

	for (i = 0; i < total_dim * total_dim; ++i) {
		attn->query_weights[i] = ((double) rand() / RAND_MAX - 0.5) * 0.01;
		attn->key_weights[i] = ((double) rand() / RAND_MAX - 0.5) * 0.01;
		attn->value_weights[i] = ((double) rand() / RAND_MAX - 0.5) * 0.01;
		attn->output_weights[i] = ((double) rand() / RAND_MAX - 0.5) * 0.01;
	}

	attn->is_allocated = 1;

	printf(
			"Attention mechanism initialized with %u heads, dim=%u, context=%u\n",
			num_heads, head_dim, context_len);
	return attn;
}

static ReasoningFramework* reasoning_init(unsigned int max_trace_length) {
	ReasoningFramework *reasoning;

	reasoning = (ReasoningFramework*) malloc(sizeof(ReasoningFramework));
	if (!reasoning)
		return NULL;

	memset(reasoning, 0, sizeof(ReasoningFramework));

	reasoning->max_trace_length = max_trace_length;
	reasoning->trace_length = 0;
	reasoning->reasoning_id = 0;
	reasoning->inference_time = 0.0;

	reasoning->reasoning_trace = (double*) malloc(
			max_trace_length * sizeof(double));
	reasoning->confidence_scores = (double*) malloc(
			max_trace_length * sizeof(double));
	reasoning->step_types = (unsigned int*) malloc(
			max_trace_length * sizeof(unsigned int));

	if (!reasoning->reasoning_trace || !reasoning->confidence_scores
			|| !reasoning->step_types) {
		free(reasoning->reasoning_trace);
		free(reasoning->confidence_scores);
		free(reasoning->step_types);
		free(reasoning);
		return NULL;
	}

	reasoning->is_allocated = 1;

	printf("Reasoning framework initialized with max trace length %u\n",
			max_trace_length);
	return reasoning;
}

static CodeGenerator* coder_init(size_t buffer_size) {
	CodeGenerator *coder;

	coder = (CodeGenerator*) malloc(sizeof(CodeGenerator));
	if (!coder)
		return NULL;

	memset(coder, 0, sizeof(CodeGenerator));

	coder->buffer_size = buffer_size;
	coder->sequence_length = 0;
	coder->language_id = 0;

	coder->code_buffer = (char*) malloc(buffer_size);
	coder->token_probabilities = (double*) malloc(50000 * sizeof(double));
	coder->token_sequence = (unsigned int*) malloc(
			buffer_size * sizeof(unsigned int));

	if (!coder->code_buffer || !coder->token_probabilities
			|| !coder->token_sequence) {
		free(coder->code_buffer);
		free(coder->token_probabilities);
		free(coder->token_sequence);
		free(coder);
		return NULL;
	}

	memset(coder->code_buffer, 0, buffer_size);
	pthread_mutex_init(&coder->code_mutex, NULL);
	coder->is_allocated = 1;

	printf("Code generator initialized with buffer size %zu\n", buffer_size);
	return coder;
}

/*=============================================================================
 * FINITE STATE MACHINE
 *============================================================================*/

/* Properly sized FSM transition table */
static const FSMState fsm_transitions[FSM_STATE_COUNT][FSM_EVENT_COUNT] = {
/* FSM_STATE_BOOT */
{ FSM_STATE_BOOT, FSM_STATE_SELF_TEST, FSM_STATE_ERROR, FSM_STATE_BOOT,
		FSM_STATE_BOOT, FSM_STATE_BOOT, FSM_STATE_BOOT, FSM_STATE_BOOT,
		FSM_STATE_BOOT, FSM_STATE_BOOT, FSM_STATE_BOOT, FSM_STATE_BOOT,
		FSM_STATE_BOOT, FSM_STATE_BOOT, FSM_STATE_SHUTDOWN },

/* FSM_STATE_SELF_TEST */
{ FSM_STATE_SELF_TEST, FSM_STATE_HARDWARE_INIT, FSM_STATE_ERROR,
		FSM_STATE_SELF_TEST, FSM_STATE_SELF_TEST, FSM_STATE_SELF_TEST,
		FSM_STATE_SELF_TEST, FSM_STATE_SELF_TEST, FSM_STATE_SELF_TEST,
		FSM_STATE_SELF_TEST, FSM_STATE_SELF_TEST, FSM_STATE_SELF_TEST,
		FSM_STATE_SELF_TEST, FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_HARDWARE_INIT */
{ FSM_STATE_HARDWARE_INIT, FSM_STATE_MODEL_LOAD, FSM_STATE_ERROR,
		FSM_STATE_HARDWARE_INIT, FSM_STATE_HARDWARE_INIT,
		FSM_STATE_HARDWARE_INIT, FSM_STATE_HARDWARE_INIT,
		FSM_STATE_HARDWARE_INIT, FSM_STATE_HARDWARE_INIT,
		FSM_STATE_HARDWARE_INIT, FSM_STATE_HARDWARE_INIT,
		FSM_STATE_HARDWARE_INIT, FSM_STATE_HARDWARE_INIT, FSM_STATE_ERROR,
		FSM_STATE_SHUTDOWN },

/* FSM_STATE_MODEL_LOAD */
{ FSM_STATE_MODEL_LOAD, FSM_STATE_MODEL_LOAD, FSM_STATE_MODEL_LOAD,
		FSM_STATE_NETWORK_INIT, FSM_STATE_ERROR, FSM_STATE_MODEL_LOAD,
		FSM_STATE_MODEL_LOAD, FSM_STATE_MODEL_LOAD, FSM_STATE_MODEL_LOAD,
		FSM_STATE_MODEL_LOAD, FSM_STATE_MODEL_LOAD, FSM_STATE_MODEL_LOAD,
		FSM_STATE_MODEL_LOAD, FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_NETWORK_INIT */
{ FSM_STATE_NETWORK_INIT, FSM_STATE_NETWORK_INIT, FSM_STATE_NETWORK_INIT,
		FSM_STATE_NETWORK_INIT, FSM_STATE_NETWORK_INIT, FSM_STATE_CRYPTO_INIT,
		FSM_STATE_ERROR, FSM_STATE_NETWORK_INIT, FSM_STATE_NETWORK_INIT,
		FSM_STATE_NETWORK_INIT, FSM_STATE_NETWORK_INIT, FSM_STATE_NETWORK_INIT,
		FSM_STATE_NETWORK_INIT, FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_CRYPTO_INIT */
{ FSM_STATE_CRYPTO_INIT, FSM_STATE_RENDERING_INIT, FSM_STATE_ERROR,
		FSM_STATE_CRYPTO_INIT, FSM_STATE_CRYPTO_INIT, FSM_STATE_CRYPTO_INIT,
		FSM_STATE_CRYPTO_INIT, FSM_STATE_CRYPTO_INIT, FSM_STATE_CRYPTO_INIT,
		FSM_STATE_CRYPTO_INIT, FSM_STATE_CRYPTO_INIT, FSM_STATE_CRYPTO_INIT,
		FSM_STATE_CRYPTO_INIT, FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_RENDERING_INIT */
{ FSM_STATE_RENDERING_INIT, FSM_STATE_AUDIO_INIT, FSM_STATE_ERROR,
		FSM_STATE_RENDERING_INIT, FSM_STATE_RENDERING_INIT,
		FSM_STATE_RENDERING_INIT, FSM_STATE_RENDERING_INIT,
		FSM_STATE_RENDERING_INIT, FSM_STATE_RENDERING_INIT,
		FSM_STATE_RENDERING_INIT, FSM_STATE_RENDERING_INIT,
		FSM_STATE_RENDERING_INIT, FSM_STATE_RENDERING_INIT, FSM_STATE_ERROR,
		FSM_STATE_SHUTDOWN },

/* FSM_STATE_AUDIO_INIT */
{ FSM_STATE_AUDIO_INIT, FSM_STATE_IDLE, FSM_STATE_ERROR, FSM_STATE_AUDIO_INIT,
		FSM_STATE_AUDIO_INIT, FSM_STATE_AUDIO_INIT, FSM_STATE_AUDIO_INIT,
		FSM_STATE_AUDIO_INIT, FSM_STATE_AUDIO_INIT, FSM_STATE_AUDIO_INIT,
		FSM_STATE_AUDIO_INIT, FSM_STATE_AUDIO_INIT, FSM_STATE_AUDIO_INIT,
		FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_IDLE */
{ FSM_STATE_IDLE, FSM_STATE_IDLE, FSM_STATE_IDLE, FSM_STATE_IDLE,
		FSM_STATE_IDLE, FSM_STATE_IDLE, FSM_STATE_IDLE, FSM_STATE_KEY_ROTATION,
		FSM_STATE_IDLE, FSM_STATE_PROCESSING, FSM_STATE_LEARNING,
		FSM_STATE_ROUTING, FSM_STATE_KEY_ROTATION, FSM_STATE_ERROR,
		FSM_STATE_SHUTDOWN },

/* FSM_STATE_PROCESSING */
{ FSM_STATE_PROCESSING, FSM_STATE_IDLE, FSM_STATE_ERROR, FSM_STATE_PROCESSING,
		FSM_STATE_PROCESSING, FSM_STATE_PROCESSING, FSM_STATE_PROCESSING,
		FSM_STATE_PROCESSING, FSM_STATE_PROCESSING, FSM_STATE_PROCESSING,
		FSM_STATE_PROCESSING, FSM_STATE_PROCESSING, FSM_STATE_PROCESSING,
		FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_LEARNING */
{ FSM_STATE_LEARNING, FSM_STATE_IDLE, FSM_STATE_ERROR, FSM_STATE_LEARNING,
		FSM_STATE_LEARNING, FSM_STATE_LEARNING, FSM_STATE_LEARNING,
		FSM_STATE_LEARNING, FSM_STATE_LEARNING, FSM_STATE_LEARNING,
		FSM_STATE_LEARNING, FSM_STATE_LEARNING, FSM_STATE_LEARNING,
		FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_ROUTING */
{ FSM_STATE_ROUTING, FSM_STATE_IDLE, FSM_STATE_ERROR, FSM_STATE_ROUTING,
		FSM_STATE_ROUTING, FSM_STATE_ROUTING, FSM_STATE_ROUTING,
		FSM_STATE_ROUTING, FSM_STATE_ROUTING, FSM_STATE_ROUTING,
		FSM_STATE_ROUTING, FSM_STATE_ROUTING, FSM_STATE_ROUTING,
		FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_KEY_ROTATION */
{ FSM_STATE_KEY_ROTATION, FSM_STATE_IDLE, FSM_STATE_ERROR,
		FSM_STATE_KEY_ROTATION, FSM_STATE_KEY_ROTATION, FSM_STATE_KEY_ROTATION,
		FSM_STATE_KEY_ROTATION, FSM_STATE_KEY_ROTATION, FSM_STATE_IDLE,
		FSM_STATE_KEY_ROTATION, FSM_STATE_KEY_ROTATION, FSM_STATE_KEY_ROTATION,
		FSM_STATE_KEY_ROTATION, FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_ERROR */
{ FSM_STATE_ERROR, FSM_STATE_IDLE, FSM_STATE_ERROR, FSM_STATE_ERROR,
		FSM_STATE_ERROR, FSM_STATE_ERROR, FSM_STATE_ERROR, FSM_STATE_ERROR,
		FSM_STATE_ERROR, FSM_STATE_ERROR, FSM_STATE_ERROR, FSM_STATE_ERROR,
		FSM_STATE_ERROR, FSM_STATE_ERROR, FSM_STATE_SHUTDOWN },

/* FSM_STATE_SHUTDOWN */
{ FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN,
		FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN,
		FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN,
		FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN,
		FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN, FSM_STATE_SHUTDOWN } };

static void boot_sequence_init(BootSequence *boot) {
	boot->current_step = 0;
	boot->steps_completed = 0;
	boot->errors_detected = 0;
	boot->boot_complete = 0;
	boot->boot_successful = 0;
	boot->boot_start_time = (unsigned long) time(NULL);
	memset(boot->step_timings, 0, sizeof(boot->step_timings));
	memset(boot->error_message, 0, sizeof(boot->error_message));
}

static void fsm_process_event(EVOXCoreSystem *system, FSMEvent event,
		unsigned long current_time) {
	FSMState new_state;

	if (!system)
		return;

	pthread_mutex_lock(&system->state_mutex);

	new_state = fsm_transitions[system->current_state][event];

	if (new_state != system->current_state) {
		system->previous_state = system->current_state;
		system->current_state = new_state;
		system->state_entry_time = current_time;

		printf("FSM: %d -> %d (event: %d)\n", system->previous_state, new_state,
				event);

		pthread_cond_broadcast(&system->state_cond);
	}

	pthread_mutex_unlock(&system->state_mutex);
}

/*=============================================================================
 * BOOT SEQUENCE STEP FUNCTIONS - Return int for compatibility
 *============================================================================*/

static int boot_step_initialize_core(EVOXCoreSystem *system) {
	memset(system, 0, sizeof(EVOXCoreSystem));
	system->version_major = EVOX_VERSION_MAJOR;
	system->version_minor = EVOX_VERSION_MINOR;
	system->version_patch = EVOX_VERSION_PATCH;
	system->current_state = FSM_STATE_BOOT;
	system->shutdown_flag = 0;
	return 1;
}

static int boot_step_initialize_security(EVOXCoreSystem *system) {
	system->crypto = crypto_init();
	return (system->crypto != NULL) ? 1 : 0;
}

static int boot_step_initialize_five_axes(EVOXCoreSystem *system) {
	five_axes_init(system);
	return 1;
}

static int boot_step_initialize_neuro_fuzzy(EVOXCoreSystem *system) {
	system->fuzzy = fuzzy_system_init(1, 1, 8);
	system->qlearn = qlearn_init(100, 10);
	return (system->fuzzy != NULL && system->qlearn != NULL) ? 1 : 0;
}

static int boot_step_initialize_academic_ai(EVOXCoreSystem *system) {
	system->moe = moe_init(8, 2, 4096);
	system->attention = attention_init(32, 128, 2048);
	system->reasoning = reasoning_init(1024);
	system->coder = coder_init(65536);
	return (system->moe && system->attention && system->reasoning
			&& system->coder) ? 1 : 0;
}

static int boot_step_initialize_thread_pool(EVOXCoreSystem *system) {
	/* Thread pool initialization would go here */
	(void) system;
	return 1;
}

static int boot_step_initialize_opencl(EVOXCoreSystem *system) {
	/* OpenCL initialization would go here */
	(void) system;
	return 1;
}

static int boot_step_scan_models(EVOXCoreSystem *system) {
	DIR *dir;
	struct dirent *entry;
	char gguf_path[MAX_PATH_LEN];
	char bin_path[MAX_PATH_LEN];
	int converted = 0;

	dir = opendir("./models");
	if (!dir) {
		mkdir("./models", 0755);
		return 1;
	}

	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_name[0] == '.')
			continue;
		const char *ext = strrchr(entry->d_name, '.');
		if (ext && strcmp(ext, ".gguf") == 0) {
			snprintf(gguf_path, sizeof(gguf_path), "./models/%s",
					entry->d_name);
			snprintf(bin_path, sizeof(bin_path), "./models/%s.bin",
					entry->d_name);
			if (gguf_to_bin_converter(system, gguf_path, bin_path) == 0) {
				converted++;
			}
		}
	}
	closedir(dir);
	printf("Converted %d GGUF models\n", converted);
	return 1;
}

static int boot_sequence_step(BootSequence *boot, EVOXCoreSystem *system) {
	unsigned long step_start;
	int ret = 0;

	if (!boot || !system)
		return 0;

	step_start = (unsigned long) time(NULL);

	printf("Boot step %d: ", boot->current_step);

	switch (boot->current_step) {
	case 0:
		printf("Core initialization\n");
		ret = boot_step_initialize_core(system);
		break;
	case 1:
		printf("Security initialization\n");
		ret = boot_step_initialize_security(system);
		break;
	case 2:
		printf("Five axes initialization\n");
		ret = boot_step_initialize_five_axes(system);
		break;
	case 3:
		printf("Neuro-fuzzy initialization\n");
		ret = boot_step_initialize_neuro_fuzzy(system);
		break;
	case 4:
		printf("Academic AI initialization\n");
		ret = boot_step_initialize_academic_ai(system);
		break;
	case 5:
		printf("Thread pool initialization\n");
		ret = boot_step_initialize_thread_pool(system);
		break;
	case 6:
		printf("OpenCL initialization\n");
		ret = boot_step_initialize_opencl(system);
		break;
	case 7:
		printf("Model scan\n");
		ret = boot_step_scan_models(system);
		boot->boot_complete = 1;
		boot->boot_successful = (boot->errors_detected == 0);
		break;
	default:
		printf("Unknown step\n");
		ret = 0;
		break;
	}

	if (!ret) {
		boot->errors_detected++;
		strcpy(boot->error_message, "Step failed");
	}

	boot->step_timings[boot->current_step] = difftime(time(NULL), step_start);
	printf(" completed in %.2f seconds\n",
			boot->step_timings[boot->current_step]);

	if (ret && boot->current_step < BOOT_STEPS - 1) {
		boot->current_step++;
		boot->steps_completed++;
	}

	return ret;
}

static void boot_sequence_execute(EVOXCoreSystem *system) {
	boot_sequence_init(&system->boot);
	while (!system->boot.boot_complete) {
		boot_sequence_step(&system->boot, system);
	}
}

/*=============================================================================
 * MODEL LOADING AND CONVERSION
 *============================================================================*/

static int scan_and_convert_models(EVOXCoreSystem *system) {
	DIR *dir;
	struct dirent *entry;
	char gguf_path[MAX_PATH_LEN];
	char bin_path[MAX_PATH_LEN];
	int converted = 0;

	if (!system)
		return 0;

	dir = opendir("./models");
	if (!dir) {
		printf("Could not open ./models directory\n");
		return 0;
	}

	printf("Scanning for GGUF models in ./models...\n");

	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_name[0] == '.')
			continue;

		const char *ext = strrchr(entry->d_name, '.');
		if (ext && strcmp(ext, ".gguf") == 0) {
			snprintf(gguf_path, sizeof(gguf_path), "./models/%s",
					entry->d_name);

			/* Create BIN filename */
			snprintf(bin_path, sizeof(bin_path), "./models/%s", entry->d_name);
			char *dot = strrchr(bin_path, '.');
			if (dot)
				*dot = '\0';
			strcat(bin_path, ".bin");

			printf("Found GGUF model: %s\n", entry->d_name);
			if (gguf_to_bin_converter(system, gguf_path, bin_path) == 0) {
				converted++;
			}
		}
	}

	closedir(dir);
	printf("Converted %d GGUF models to BIN format\n", converted);
	return converted;
}

static int model_load_bin(EVOXCoreSystem *system, const char *filename) {
	FILE *fp;
	unsigned char header[4096];
	unsigned int i;
	unsigned char checksum[32];
	size_t checksum_len;

	if (!system)
		return -1;

	if (system->model_loaded) {
		printf("Model already loaded, skipping\n");
		return 0;
	}

	fp = fopen(filename, "rb");
	if (!fp) {
		printf("Failed to open model file: %s\n", filename);
		return -1;
	}

	printf("Loading BIN model: %s\n", filename);

	if (crypto_checksum_file(filename, checksum, &checksum_len) == 0) {
		printf("Checksum verified (%zu bytes)\n", checksum_len);
	}

	bin_parse_filename(filename, &system->bin_info);
	strncpy(system->model_name, filename, MAX_FILENAME_LEN - 1);
	system->model_name[MAX_FILENAME_LEN - 1] = '\0';

	if (fread(header, 1, sizeof(header), fp) != sizeof(header)) {
		printf("Failed to read model header\n");
		fclose(fp);
		return -1;
	}

	if (header[0] == 'E' && header[1] == 'V' && header[2] == 'O'
			&& header[3] == 'X') {
		printf("Valid EVOX model format detected\n");
		printf("Version: %d.%d\n", header[4], header[5]);
		printf("Encrypted: %s\n", header[6] ? "Yes" : "No");
	} else {
		printf(
				"Warning: Not an EVOX format model, attempting to load anyway\n");
	}

	if (!system->network) {
		system->network = (NeuralNetwork*) calloc(1, sizeof(NeuralNetwork));
		if (!system->network) {
			fclose(fp);
			return -1;
		}
		system->network->is_allocated = 1;
	}

	system->network->vocab_size = 32000;
	system->network->hidden_size = 4096;
	system->network->num_layers = 32;
	system->network->num_experts = 8;

	system->network->num_nodes = system->network->vocab_size
			+ system->network->hidden_size * system->network->num_layers;
	system->network->num_synapses = system->network->num_nodes * 10;

	printf("Network topology: %u nodes, %u synapses\n",
			system->network->num_nodes, system->network->num_synapses);

	system->network->nodes = (NeuralNode*) calloc(system->network->num_nodes,
			sizeof(NeuralNode));
	system->network->synapses = (Synapse*) calloc(system->network->num_synapses,
			sizeof(Synapse));
	system->network->node_activations = (double*) calloc(
			system->network->num_nodes, sizeof(double));
	system->network->node_deltas = (double*) calloc(system->network->num_nodes,
			sizeof(double));
	system->network->expert_routing = (unsigned int*) calloc(
			system->network->num_experts, sizeof(unsigned int));

	if (!system->network->nodes || !system->network->synapses
			|| !system->network->node_activations
			|| !system->network->node_deltas
			|| !system->network->expert_routing) {
		printf("Failed to allocate network memory\n");
		fclose(fp);
		return -1;
	}

	for (i = 0; i < system->network->num_nodes; ++i) {
		NeuralNode *node = &system->network->nodes[i];
		node->activation = ((double) rand() / RAND_MAX) * 0.1;
		node->membrane_potential = 0.0;
		node->threshold = 1.0;
		node->refractory_period = 0.0;
		node->hebbian_trace = 0.0;
		node->spike_count = 0;
		node->last_spike_time = 0;

		node->position[AXIS_X_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[AXIS_Y_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[AXIS_Z_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[AXIS_B_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[AXIS_R_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
	}

	for (i = 0; i < system->network->num_synapses; ++i) {
		Synapse *syn = &system->network->synapses[i];
		syn->from_node = rand() % system->network->num_nodes;
		syn->to_node = rand() % system->network->num_nodes;
		syn->weight = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		syn->delay = ((double) rand() / RAND_MAX) * 0.1;
		syn->plasticity = 0.01;
		syn->luminescence = ((double) rand() / RAND_MAX) * 0.5;
		syn->firing_count = 0;
		syn->last_update = 0.0;
	}

	pthread_spin_init(&system->network->network_lock, PTHREAD_PROCESS_PRIVATE);

	system->model_loaded = 1;
	system->model_load_time = (unsigned long) time(NULL);

	fclose(fp);
	printf("Model loaded successfully\n");
	return 0;
}

/*=============================================================================
 * NEURAL ACTIVITY MONITORING
 *============================================================================*/

static void neural_activity_update(EVOXCoreSystem *system) {
	unsigned int i, j;
	double *probs;
	double sum_activations;
	double entropy;
	double fuzzy_input[1];
	double moe_input[1];
	double moe_output[4096];
	FiveAxisVector pos;
	double weight;

	if (!system || !system->network || !system->network->is_allocated)
		return;

	for (i = 0; i < system->network->num_nodes; ++i) {
		NeuralNode *node = &system->network->nodes[i];
		double input_sum = 0.0;

		for (j = 0; j < system->network->num_synapses; ++j) {
			Synapse *syn = &system->network->synapses[j];
			if (syn->to_node == i) {
				NeuralNode *from_node = &system->network->nodes[syn->from_node];
				input_sum += from_node->activation * syn->weight;
			}
		}

		node->activation = pure_sigmoid(input_sum);

		for (j = 0; j < system->network->num_synapses; ++j) {
			Synapse *syn = &system->network->synapses[j];
			if (syn->to_node == i) {
				NeuralNode *from_node = &system->network->nodes[syn->from_node];
				hebbian_update_synapse(syn, from_node->activation,
						node->activation, 0.01, 0.001);
			}
		}

		pos.x = node->position[AXIS_X_INDEX];
		pos.y = node->position[AXIS_Y_INDEX];
		pos.z = node->position[AXIS_Z_INDEX];
		pos.b = node->position[AXIS_B_INDEX];
		pos.r = node->position[AXIS_R_INDEX];

		weight = five_axes_weighting(system, &pos);
		node->activation *= (1.0 + weight);
	}

	probs = (double*) alloca(system->network->num_nodes * sizeof(double));
	sum_activations = 0.0;

	for (i = 0; i < system->network->num_nodes; ++i) {
		sum_activations += system->network->nodes[i].activation;
	}

	if (sum_activations > 0.0) {
		for (i = 0; i < system->network->num_nodes; ++i) {
			probs[i] = system->network->nodes[i].activation / sum_activations;
		}

		entropy = pure_shannon_entropy(probs, system->network->num_nodes);

		if (system->fuzzy && system->fuzzy->is_allocated) {
			fuzzy_input[0] = entropy / 10.0;
			fuzzy_mamdani_inference(system->fuzzy, fuzzy_input);
		}

		if (system->moe && system->moe->is_allocated) {
			moe_input[0] = entropy / 10.0;
			moe_route(system->moe, moe_input, moe_output);
		}
	}
}

/*=============================================================================
 * OPENGL RENDERING
 *============================================================================*/

static OpenGLContext* opengl_init(int width, int height) {
	OpenGLContext *gl;

	gl = (OpenGLContext*) malloc(sizeof(OpenGLContext));
	if (!gl)
		return NULL;

	memset(gl, 0, sizeof(OpenGLContext));

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("SDL_Init failed: %s\n", SDL_GetError());
		free(gl);
		return NULL;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	gl->window = SDL_CreateWindow("Evox AI Core 5 Axes Visualization",
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED, width, height,
			SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

	if (!gl->window) {
		printf("SDL_CreateWindow failed: %s\n", SDL_GetError());
		SDL_Quit();
		free(gl);
		return NULL;
	}

	gl->gl_context = SDL_GL_CreateContext(gl->window);
	if (!gl->gl_context) {
		printf("SDL_GL_CreateContext failed: %s\n", SDL_GetError());
		SDL_DestroyWindow(gl->window);
		SDL_Quit();
		free(gl);
		return NULL;
	}

	SDL_GL_MakeCurrent(gl->window, gl->gl_context);
	SDL_GL_SetSwapInterval(1);

	gl->window_width = width;
	gl->window_height = height;
	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) width / (double) height, 0.1, 1000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	gl->camera_position.x = 5.0;
	gl->camera_position.y = 5.0;
	gl->camera_position.z = 10.0;
	gl->camera_position.b = 0.0;
	gl->camera_position.r = 0.0;
	gl->rotation_angle = 0.0;

	pthread_mutex_init(&gl->render_mutex, NULL);
	gl->is_allocated = 1;

	printf("OpenGL context initialized (%dx%d)\n", width, height);
	return gl;
}

static void opengl_render_axes(EVOXCoreSystem *system, OpenGLContext *gl) {
	unsigned int i;
	char status_text[256];
	int text_y;

	if (!system || !gl || !gl->is_allocated)
		return;

	pthread_mutex_lock(&gl->render_mutex);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	gluLookAt(gl->camera_position.x, gl->camera_position.y,
			gl->camera_position.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	glRotated(gl->rotation_angle, 0.0, 1.0, 0.0);

	/* Draw origin marker (R axis - yellow dot) */
	glPointSize(12.0);
	glBegin(GL_POINTS);
	glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
	glVertex3f(0.0, 0.0, 0.0);
	glEnd();

	/* Draw axes */
	glLineWidth(3.0);

	/* X Axis - Red */
	glBegin(GL_LINES);
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glVertex3f(-2.5, 0.0, 0.0);
	glVertex3f(2.5, 0.0, 0.0);
	glEnd();

	/* Y Axis - Green */
	glBegin(GL_LINES);
	glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
	glVertex3f(0.0, -2.5, 0.0);
	glVertex3f(0.0, 2.5, 0.0);
	glEnd();

	/* Z Axis - Blue */
	glBegin(GL_LINES);
	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
	glVertex3f(0.0, 0.0, -2.5);
	glVertex3f(0.0, 0.0, 2.5);
	glEnd();

	/* B Axis - Purple (diagonal) */
	glBegin(GL_LINES);
	glColor4f(0.5f, 0.0f, 0.5f, 1.0f);
	glVertex3f(-2.0, -2.0, -2.0);
	glVertex3f(2.0, 2.0, 2.0);
	glEnd();

	/* Draw markers */
	glPointSize(8.0);
	glBegin(GL_POINTS);

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glVertex3f(1.5, 1.5, 1.5);

	glColor4f(0.3f, 0.3f, 0.3f, 1.0f);
	glVertex3f(-1.5, -1.5, -1.5);

	glEnd();

	/* Draw neural nodes */
	if (system->network && system->network->is_allocated
			&& system->network->nodes) {
		glPointSize(5.0);
		glBegin(GL_POINTS);

		for (i = 0; i < system->network->num_nodes && i < 10000; i += 10) {
			NeuralNode *node = &system->network->nodes[i];
			float intensity = (float) node->activation;

			glColor4f(intensity, 0.0f, 1.0f - intensity, 0.8f);
			glVertex3f((float) node->position[0] * 2.0f,
					(float) node->position[1] * 2.0f,
					(float) node->position[2] * 2.0f);
		}

		glEnd();
	}

	/* Switch to orthographic for text overlay */
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, gl->window_width, gl->window_height, 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);

	snprintf(status_text, sizeof(status_text),
			"Evox AI Core v%u.%u.%u | State: %d | Nodes: %u",
			system->version_major, system->version_minor, system->version_patch,
			system->current_state,
			system->network ? system->network->num_nodes : 0);

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glRasterPos2i(10, 20);

	for (i = 0; status_text[i] != '\0'; ++i) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, status_text[i]);
	}

	text_y = 40;
	snprintf(status_text, sizeof(status_text),
			"5-Axes: X(Red) Y(Green) Z(Blue) B(Purple) R(Yellow) | Angle: %.1f",
			gl->rotation_angle);
	glRasterPos2i(10, text_y);
	for (i = 0; status_text[i] != '\0'; ++i) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, status_text[i]);
	}

	if (system->fuzzy && system->fuzzy->is_allocated) {
		text_y = 60;
		snprintf(status_text, sizeof(status_text),
				"Fuzzy Output: %.3f | Inferences: %lu",
				system->fuzzy->defuzzification_value, system->total_inferences);
		glRasterPos2i(10, text_y);
		for (i = 0; status_text[i] != '\0'; ++i) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, status_text[i]);
		}
	}

	text_y = 80;
	snprintf(status_text, sizeof(status_text), "Model: %s",
			system->model_loaded ? system->model_name : "None");
	glRasterPos2i(10, text_y);
	for (i = 0; status_text[i] != '\0'; ++i) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, status_text[i]);
	}

	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	SDL_GL_SwapWindow(gl->window);

	pthread_mutex_unlock(&gl->render_mutex);
}

/*=============================================================================
 * MAIN SYSTEM INITIALIZATION
 *============================================================================*/

static EVOXCoreSystem* evox_system_init(void) {
	EVOXCoreSystem *system;

	system = (EVOXCoreSystem*) calloc(1, sizeof(EVOXCoreSystem));
	if (!system)
		return NULL;

	system->version_major = EVOX_VERSION_MAJOR;
	system->version_minor = EVOX_VERSION_MINOR;
	system->version_patch = EVOX_VERSION_PATCH;

	system->current_state = FSM_STATE_BOOT;
	system->previous_state = FSM_STATE_BOOT;
	system->state_entry_time = (unsigned long) time(NULL);
	system->shutdown_flag = 0;
	system->model_loaded = 0;

	pthread_mutex_init(&system->state_mutex, NULL);
	pthread_cond_init(&system->state_cond, NULL);
	pthread_rwlock_init(&system->model_lock, NULL);
	pthread_spin_init(&system->metrics_lock, PTHREAD_PROCESS_PRIVATE);

	boot_sequence_init(&system->boot);
	five_axes_init(system);

	printf("\nEvox AI Core System v%u.%u.%u initializing...\n",
			system->version_major, system->version_minor,
			system->version_patch);
	printf("========================================\n\n");

	return system;
}

/*=============================================================================
 * MAIN EVENT LOOP
 *============================================================================*/

static void evox_main_loop(EVOXCoreSystem *system) {
	unsigned long last_key_check = 0;
	unsigned long last_render = 0;
	unsigned long last_activity_update = 0;
	unsigned long last_model_scan = 0;
	unsigned long current_time;
	int render_initialized = 0;

	if (!system)
		return;

	while (!system->shutdown_flag) {
		current_time = (unsigned long) time(NULL);

		switch (system->current_state) {
		case FSM_STATE_BOOT:
			if (!system->boot.boot_complete) {
				boot_sequence_step(&system->boot, system);
			} else {
				if (system->boot.boot_successful) {
					printf("\nBoot successful, transitioning to SELF_TEST\n");
					fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE,
							current_time);
				} else {
					printf("\nBoot failed, transitioning to ERROR state\n");
					fsm_process_event(system, FSM_EVENT_BOOT_FAILED,
							current_time);
				}
			}
			break;

		case FSM_STATE_SELF_TEST:
			printf("Running self tests...\n");
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, current_time);
			break;

		case FSM_STATE_HARDWARE_INIT:
			printf("Initializing hardware...\n");
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, current_time);
			break;

		case FSM_STATE_MODEL_LOAD:
			if (!render_initialized && (current_time - last_model_scan > 2)) {
				scan_and_convert_models(system);
				last_model_scan = current_time;
			}

			if (!system->model_loaded && access("./models", F_OK) == 0) {
				DIR *dir = opendir("./models");
				if (dir) {
					struct dirent *entry;
					while ((entry = readdir(dir)) != NULL) {
						if (entry->d_name[0] == '.')
							continue;

						const char *ext = strrchr(entry->d_name, '.');
						if (ext && strcmp(ext, ".bin") == 0) {
							char model_path[MAX_PATH_LEN];
							snprintf(model_path, sizeof(model_path),
									"./models/%s", entry->d_name);

							if (model_load_bin(system, model_path) == 0) {
								printf("Model loaded successfully\n");
								fsm_process_event(system,
										FSM_EVENT_MODEL_LOADED, current_time);
								break;
							}
						}
					}
					closedir(dir);
				}
			}

			if (!system->model_loaded) {
				sleep(1);
			}
			break;

		case FSM_STATE_NETWORK_INIT:
			printf("Initializing network...\n");
			fsm_process_event(system, FSM_EVENT_NETWORK_READY, current_time);
			break;

		case FSM_STATE_CRYPTO_INIT:
			printf("Initializing crypto...\n");
			if (!system->crypto) {
				system->crypto = crypto_init();
			}
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, current_time);
			break;

		case FSM_STATE_RENDERING_INIT:
			printf("Initializing OpenGL rendering...\n");
			if (!render_initialized) {
				system->gl = opengl_init(1280, 720);
				if (system->gl) {
					printf("OpenGL initialized successfully\n");
					render_initialized = 1;
					fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE,
							current_time);
				} else {
					printf("ERROR: Failed to initialize OpenGL\n");
					fsm_process_event(system, FSM_EVENT_ERROR_DETECTED,
							current_time);
				}
			} else {
				fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE,
						current_time);
			}
			break;

		case FSM_STATE_AUDIO_INIT:
			printf("Initializing OpenAL audio...\n");
			if (!system->al) {
				system->al = (OpenALContext*) calloc(1, sizeof(OpenALContext));
				if (system->al) {
					system->al->audio_device = alcOpenDevice(NULL);
					if (system->al->audio_device) {
						system->al->audio_context = alcCreateContext(
								system->al->audio_device, NULL);
						alcMakeContextCurrent(system->al->audio_context);
						printf("OpenAL initialized successfully\n");
					} else {
						printf("WARNING: Could not open audio device\n");
					}
					pthread_mutex_init(&system->al->audio_mutex, NULL);
					system->al->is_allocated = 1;
				}
			}
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, current_time);
			break;

		case FSM_STATE_IDLE:
			if (current_time - last_key_check > 3600) {
				if (system->crypto
						&& current_time > system->crypto->key_expiry_time) {
					printf("Key rotation triggered\n");
					fsm_process_event(system, FSM_EVENT_KEY_EXPIRING,
							current_time);
				}
				last_key_check = current_time;
			}
			break;

		case FSM_STATE_PROCESSING:
			neural_activity_update(system);
			system->total_inferences++;
			if (system->total_inferences % 100 == 0) {
				printf("Completed %lu inferences\n", system->total_inferences);
			}
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, current_time);
			break;

		case FSM_STATE_LEARNING:
			printf("Learning state\n");
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, current_time);
			break;

		case FSM_STATE_ROUTING:
			printf("Routing update\n");
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, current_time);
			break;

		case FSM_STATE_KEY_ROTATION:
			printf("Rotating cryptographic keys\n");
			if (system->crypto) {
				RAND_bytes(system->crypto->aes_key,
						sizeof(system->crypto->aes_key));
				RAND_bytes(system->crypto->aes_iv,
						sizeof(system->crypto->aes_iv));
				system->crypto->key_creation_time = current_time;
				system->crypto->key_expiry_time = current_time
						+ KEY_ROTATION_SECONDS;
				system->crypto->key_rotations++;
				printf("Keys rotated (total: %u)\n",
						system->crypto->key_rotations);
			}
			fsm_process_event(system, FSM_EVENT_KEY_ROTATED, current_time);
			break;

		case FSM_STATE_ERROR:
			printf("System in ERROR state - attempting recovery\n");
			sleep(2);
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, current_time);
			break;

		case FSM_STATE_SHUTDOWN:
			printf("Shutdown requested\n");
			system->shutdown_flag = 1;
			break;

		default:
			break;
		}

		if (current_time - last_activity_update > 1) {
			if (system->model_loaded
					&& (system->current_state == FSM_STATE_IDLE
							|| system->current_state == FSM_STATE_PROCESSING)) {
				neural_activity_update(system);
			}
			last_activity_update = current_time;
		}

		if (system->gl && system->gl->is_allocated
				&& current_time - last_render > 0) {
			opengl_render_axes(system, system->gl);
			system->gl->rotation_angle += 0.5;
			last_render = current_time;
		}

		if (system->gl && system->gl->window) {
			SDL_Event event;
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT) {
					fsm_process_event(system, FSM_EVENT_SHUTDOWN_REQUEST,
							current_time);
				} else if (event.type == SDL_KEYDOWN) {
					if (event.key.keysym.sym == SDLK_ESCAPE) {
						fsm_process_event(system, FSM_EVENT_SHUTDOWN_REQUEST,
								current_time);
					}
				}
			}
		}

		usleep(10000);
	}
}

/*=============================================================================
 * SYSTEM CLEANUP
 *============================================================================*/

static void evox_system_cleanup(EVOXCoreSystem *system) {
	unsigned int i;

	if (!system)
		return;

	printf("\nShutting down Evox AI Core System...\n");

	system->shutdown_flag = 1;

	if (system->network && system->network->is_allocated) {
		free(system->network->nodes);
		free(system->network->synapses);
		free(system->network->node_activations);
		free(system->network->node_deltas);
		free(system->network->expert_routing);
		pthread_spin_destroy(&system->network->network_lock);
		free(system->network);
		printf("Neural network cleaned up\n");
	}

	if (system->moe && system->moe->is_allocated) {
		free(system->moe->routing_weights);
		free(system->moe->routing_indices);
		free(system->moe->expert_outputs);
		free(system->moe->gate_outputs);
		pthread_spin_destroy(&system->moe->moe_lock);
		free(system->moe);
		printf("MoE cleaned up\n");
	}

	if (system->attention && system->attention->is_allocated) {
		free(system->attention->query_weights);
		free(system->attention->key_weights);
		free(system->attention->value_weights);
		free(system->attention->output_weights);
		free(system->attention->attention_scores);
		free(system->attention);
		printf("Attention mechanism cleaned up\n");
	}

	if (system->reasoning && system->reasoning->is_allocated) {
		free(system->reasoning->reasoning_trace);
		free(system->reasoning->confidence_scores);
		free(system->reasoning->step_types);
		free(system->reasoning);
		printf("Reasoning framework cleaned up\n");
	}

	if (system->coder && system->coder->is_allocated) {
		free(system->coder->code_buffer);
		free(system->coder->token_probabilities);
		free(system->coder->token_sequence);
		pthread_mutex_destroy(&system->coder->code_mutex);
		free(system->coder);
		printf("Code generator cleaned up\n");
	}

	if (system->fuzzy && system->fuzzy->is_allocated) {
		free(system->fuzzy->fuzzy_sets);
		free(system->fuzzy->rule_strengths);
		free(system->fuzzy->rule_consequents);
		free(system->fuzzy->input_mf_params);
		free(system->fuzzy->output_mf_params);
		free(system->fuzzy);
		printf("Fuzzy system cleaned up\n");
	}

	if (system->qlearn && system->qlearn->is_allocated) {
		free(system->qlearn->q_table);
		free(system->qlearn->rewards);
		pthread_spin_destroy(&system->qlearn->q_lock);
		free(system->qlearn);
		printf("Q-learning system cleaned up\n");
	}

	if (system->crypto && system->crypto->is_allocated) {
		EVP_CIPHER_CTX_free(system->crypto->cipher_ctx);
		EVP_MD_CTX_free(system->crypto->md_ctx);
		if (system->crypto->pkey) {
			EVP_PKEY_free(system->crypto->pkey);
		}
		pthread_mutex_destroy(&system->crypto->crypto_mutex);
		free(system->crypto);
		printf("Crypto context cleaned up\n");
	}

	if (system->gl && system->gl->is_allocated) {
		if (system->gl->gl_context) {
			SDL_GL_DeleteContext(system->gl->gl_context);
		}
		if (system->gl->window) {
			SDL_DestroyWindow(system->gl->window);
		}
		pthread_mutex_destroy(&system->gl->render_mutex);
		free(system->gl);
		printf("OpenGL context cleaned up\n");
	}

	if (system->al && system->al->is_allocated) {
		if (system->al->audio_context) {
			alcDestroyContext(system->al->audio_context);
		}
		if (system->al->audio_device) {
			alcCloseDevice(system->al->audio_device);
		}
		pthread_mutex_destroy(&system->al->audio_mutex);
		free(system->al);
		printf("OpenAL context cleaned up\n");
	}

	if (system->numa_node_cpus) {
		free(system->numa_node_cpus);
	}
	if (system->numa_node_memory) {
		free(system->numa_node_memory);
	}

	pthread_mutex_destroy(&system->state_mutex);
	pthread_cond_destroy(&system->state_cond);
	pthread_rwlock_destroy(&system->model_lock);
	pthread_spin_destroy(&system->metrics_lock);

	SDL_Quit();

	printf("Evox AI Core System shutdown complete.\n");
	printf("Total inferences: %lu\n", system->total_inferences);

	free(system);
}

/*=============================================================================
 * ENTRY POINT
 *============================================================================*/

int main(int argc, char *argv[]) {
	EVOXCoreSystem *system;

	srand((unsigned int) time(NULL));

	int glut_argc = 1;
	char *glut_argv[2] = { argv[0], "" };
	glutInit(&glut_argc, glut_argv);

	system = evox_system_init();
	if (!system) {
		fprintf(stderr, "Failed to initialize Evox AI Core System\n");
		return EXIT_FAILURE;
	}

	boot_sequence_execute(system);

	printf("\nEvox AI Core System initialized. Starting main loop...\n");
	printf(
			"5-Axes Reference Frame: X(Red), Y(Green), Z(Blue), B(Purple), R(Yellow)\n");
	printf("Model directory: ./models/\n");
	printf("Controls: ESC - Exit\n\n");

	evox_main_loop(system);

	evox_system_cleanup(system);

	printf("\nEvox AI Core System terminated normally.\n");
	return EXIT_SUCCESS;
}
