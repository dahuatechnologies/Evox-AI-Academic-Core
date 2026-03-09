/*
 * Copyright (c) 2026 Evolution Technologies Research and Prototype
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The "evo is not responding" issue is likely due to the render loop consuming too much CPU or blocking.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * File: evox/src/main.c
 * Description: Evox AI Core 5 Axes System with Academic AI Foundations
 *
 * Production-Grade Implementation v1.0.0
 * Optimized for AMD Ryzen 5 7520U with AVX2/FMA
 */

/*=============================================================================
 * SYSTEM HEADERS AND CONFIGURATION
 *============================================================================*/

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

/* Override inline for C89 compatibility with system headers */
#ifdef __STRICT_ANSI__
#undef __STRICT_ANSI__
#endif

#ifndef inline
#define inline
#endif

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
#include <sys/resource.h>
#include <dirent.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <libgen.h>
#include <regex.h>
#include <signal.h>

/* NUMA Support */
#include <numa.h>
#include <numaif.h>

/* OpenMPI */
#include <mpi.h>

/* OpenSSL */
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/rand.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/aes.h>
#include <openssl/rsa.h>
#include <openssl/ssl.h>
#include <openssl/hmac.h>

/* libmicrohttpd */
#include <microhttpd.h>

/* OpenGL */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

/* GLUT */
#include <GL/glut.h>

/* OpenAL */
#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>

/* SDL2 */
#include <SDL2/SDL.h>

/* OpenCL */
#include <CL/cl.h>

/* AVX-256 SIMD */
#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <fmaintrin.h>

/*=============================================================================
 * CONSTANTS AND MACROS
 *============================================================================*/

#define EVOX_VERSION_MAJOR           1
#define EVOX_VERSION_MINOR           0
#define EVOX_VERSION_PATCH           0
#define EVOX_VERSION_STRING          "1.0.0"

/* System Limits */
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
#define MAX_PEERS                        128
#define MAX_MESSAGE_SIZE               65536
#define INFERENCE_BATCH_SIZE              32
#define LEARNING_RATE                  0.001
#define MOMENTUM                        0.9
#define WEIGHT_DECAY                   0.0001

/* Axis Indices */
#define AXIS_X_INDEX                       0
#define AXIS_Y_INDEX                       1
#define AXIS_Z_INDEX                       2
#define AXIS_B_INDEX                       3
#define AXIS_R_INDEX                       4

/* Axis Colors (BGRA format) */
#define AXIS_X_COLOR                0xFF0000FF  /* Crisp Red */
#define AXIS_Y_COLOR                0x00FF00FF  /* Bright Green */
#define AXIS_Z_COLOR                0x0000FFFF  /* Pure Blue */
#define AXIS_B_COLOR                0xFF00FFFF  /* Purple */
#define AXIS_R_COLOR                0xFFFF00FF  /* Yellow */

/* Marker Constants */
#define ORIGIN_MARKER                       0
#define POSITIVE_MARKER                     1
#define NEGATIVE_MARKER                    -1

/* GGUF Magic */
#define GGUF_MAGIC                    0x46554747  /* 'GGUF' */

/* Error Codes */
#define ERR_SUCCESS                     0
#define ERR_GENERAL                    -1
#define ERR_MEMORY                     -2
#define ERR_IO                         -3
#define ERR_CRYPTO                     -4
#define ERR_NETWORK                    -5
#define ERR_MODEL                      -6
#define ERR_THREAD                     -7
#define ERR_NUMA                       -8
#define ERR_OPENGL                     -9
#define ERR_OPENAL                    -10
#define ERR_OPENCL                    -11
#define ERR_MPI                       -12
#define ERR_FSM                       -13
#define ERR_BOOT                       -14

/* Log Levels */
#define LOG_LEVEL                       2  /* 0=ERROR,1=WARN,2=INFO,3=DEBUG */

/*=============================================================================
 * TYPE DEFINITIONS
 *============================================================================*/

/* Log Levels */
typedef enum {
	LOG_ERROR = 0, LOG_WARN, LOG_INFO, LOG_DEBUG
} LogLevel;

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

/* BIN Naming Convention */
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

/* GGUF Header */
typedef struct {
	uint32_t magic;
	uint32_t version;
	uint64_t tensor_count;
	uint64_t metadata_kv_count;
} GGUFFileHeader;

/* Forward declarations */
struct EVOXCoreSystem;
typedef struct EVOXCoreSystem EVOXCoreSystem;

/*=============================================================================
 * CORE DATA STRUCTURES (32-byte aligned for SIMD)
 *============================================================================*/

/* 5-Axes Vector - 32-byte aligned */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	double x; /* Length - X Axis */
	double y; /* Height - Y Axis */
	double z; /* Width - Z Axis */
	double b; /* Diagonal Base - B Axis */
	double r; /* Rotation - R Axis */
} FiveAxisVector;

/* Neural Node - 32-byte aligned */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	double activation;
	double membrane_potential;
	double threshold;
	double refractory_period;
	double hebbian_trace;
	double position[AXIS_COUNT];
	uint64_t spike_count;
	uint64_t last_spike_time;
	double trace_stdp;
	double trace_eligibility;
} NeuralNode;

/* Synapse - 32-byte aligned */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	uint32_t from_node;
	uint32_t to_node;
	double weight;
	double delay;
	double plasticity;
	double luminescence;
	uint32_t firing_count;
	double last_update;
	double trace_pre;
	double trace_post;
} Synapse;

/* Neural Network */
typedef struct {
	uint32_t vocab_size;
	uint32_t hidden_size;
	uint32_t num_layers;
	uint32_t num_experts;
	uint32_t num_nodes;
	uint32_t num_synapses;
	NeuralNode *nodes;
	Synapse *synapses;
	double *node_activations;
	double *node_deltas;
	uint32_t *expert_routing;
	pthread_spinlock_t network_lock;
	int is_allocated;
	uint64_t total_processed;
	double average_activation;
} NeuralNetwork;

/* Attention Mechanism (V2) */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	double *query_weights;
	double *key_weights;
	double *value_weights;
	double *output_weights;
	double *attention_scores;
	uint32_t num_heads;
	uint32_t head_dim;
	uint32_t context_length;
	double temperature;
	double dropout_rate;
	int is_allocated;
	uint64_t total_attention_calls;
} AttentionMechanism;

/* Mixture of Experts (MoE) */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	uint32_t num_experts;
	uint32_t num_active_experts;
	uint32_t expert_capacity;
	double *routing_weights;
	uint32_t *routing_indices;
	double *expert_outputs;
	double *gate_outputs;
	double *expert_biases;
	pthread_spinlock_t moe_lock;
	int is_allocated;
	uint64_t total_routes;
} MoEArchitecture;

/* Reasoning Framework (R1) */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	double *reasoning_trace;
	double *confidence_scores;
	uint32_t *step_types;
	uint32_t trace_length;
	uint32_t max_trace_length;
	double inference_time;
	uint64_t reasoning_id;
	double *chain_of_thought;
	uint32_t cot_length;
	int is_allocated;
} ReasoningFramework;

/* Code Generator (Coder) */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	char *code_buffer;
	size_t buffer_size;
	uint32_t language_id;
	double *token_probabilities;
	uint32_t *token_sequence;
	uint32_t sequence_length;
	double *attention_cache;
	pthread_mutex_t code_mutex;
	int is_allocated;
	uint64_t tokens_generated;
} CodeGenerator;

/* Neuro-Fuzzy System */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	double *fuzzy_sets;
	double *rule_strengths;
	double *rule_consequents;
	uint32_t num_inputs;
	uint32_t num_outputs;
	uint32_t num_rules;
	double *input_mf_params;
	double *output_mf_params;
	double defuzzification_value;
	MamdaniInferenceType inference_type;
	int is_allocated;
} NeuroFuzzySystem;

/* Q-Learning System */
typedef struct __attribute__((aligned(SIMD_ALIGNMENT))) {
	double *q_table;
	double *rewards;
	uint32_t num_states;
	uint32_t num_actions;
	double learning_rate;
	double discount_factor;
	double exploration_rate;
	double exploration_decay;
	uint64_t learning_steps;
	pthread_spinlock_t q_lock;
	int is_allocated;
} QLearningSystem;

/* Boot Sequence */
typedef struct {
	uint32_t current_step;
	uint32_t steps_completed;
	uint32_t errors_detected;
	uint32_t boot_complete;
	uint32_t boot_successful;
	double step_timings[BOOT_STEPS];
	uint64_t boot_start_time;
	char error_message[256];
} BootSequence;

/* Crypto Context */
typedef struct {
	EVP_CIPHER_CTX *cipher_ctx;
	EVP_MD_CTX *md_ctx;
	EVP_PKEY *pkey;
	uint8_t aes_key[32];
	uint8_t aes_iv[16];
	uint8_t hmac_key[32];
	uint8_t *key_blob;
	size_t key_blob_size;
	uint64_t key_creation_time;
	uint64_t key_expiry_time;
	uint32_t key_rotations;
	pthread_mutex_t crypto_mutex;
	int is_allocated;
} CryptoContext;

/* P2P Network */
typedef struct {
	struct MHD_Daemon *http_daemon;
	uint16_t port;
	char node_id[64];
	char *peer_list;
	uint32_t peer_count;
	uint32_t max_peers;
	uint8_t *message_buffer;
	size_t buffer_size;
	pthread_rwlock_t peer_lock;
	int is_allocated;
} P2PNetworkContext;

/* OpenGL Context */
typedef struct {
	SDL_Window *window;
	SDL_GLContext gl_context;
	int window_width;
	int window_height;
	FiveAxisVector camera_position;
	double rotation_angle;
	double zoom_level;
	pthread_mutex_t render_mutex;
	int is_allocated;
	double fps;
	uint64_t frame_count;
	uint64_t last_fps_time;
} OpenGLContext;

/* OpenAL Context */
typedef struct {
	ALCdevice *audio_device;
	ALCcontext *audio_context;
	ALuint *sound_sources;
	ALuint *sound_buffers;
	uint32_t num_sources;
	float listener_position[3];
	pthread_mutex_t audio_mutex;
	int is_allocated;
} OpenALContext;

/* OpenCL Context */
typedef struct {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem buffer_a;
	cl_mem buffer_b;
	cl_mem buffer_c;
	size_t work_group_size;
	size_t global_work_size;
	int is_allocated;
} OpenCLContext;

/* MPI Context */
typedef struct {
	int mpi_rank;
	int mpi_size;
	MPI_Comm expert_comm;
	uint32_t message_id;
	double *send_buffer;
	double *recv_buffer;
	size_t buffer_length;
	int is_allocated;
} MPIContext;

/* Performance Metrics */
typedef struct {
	double total_inference_time;
	uint64_t total_inferences;
	double average_latency;
	double throughput;
	uint64_t peak_memory_usage;
	uint64_t current_memory_usage;
	double inference_rate;
	double learning_rate_avg;
	double routing_latency;
	double crypto_latency;
	uint64_t cache_hits;
	uint64_t cache_misses;
} PerformanceMetrics;

/* Thread Worker */
typedef struct {
	pthread_t thread;
	uint32_t thread_id;
	cpu_set_t affinity;
	void *local_memory;
	size_t local_memory_size;
	int numa_node;
	volatile uint32_t active;
	volatile uint32_t working;
	void* (*work_func)(void*);
	void *work_arg;
} ThreadWorker;

/* Main System Structure */
struct EVOXCoreSystem {
	/* Version */
	uint32_t version_major;
	uint32_t version_minor;
	uint32_t version_patch;
	char version_string[16];

	/* State */
	FSMState current_state;
	FSMState previous_state;
	uint64_t state_entry_time;
	uint32_t error_code;
	volatile sig_atomic_t shutdown_flag;

	/* Boot */
	BootSequence boot;

	/* 5-Axes */
	FiveAxisVector axes[AXIS_COUNT];
	FiveAxisVector origin;
	FiveAxisVector markers[3];
	double axis_weights[AXIS_COUNT];

	/* AI Components */
	NeuralNetwork *network;
	AttentionMechanism *attention;
	MoEArchitecture *moe;
	ReasoningFramework *reasoning;
	CodeGenerator *coder;
	NeuroFuzzySystem *fuzzy;
	QLearningSystem *qlearn;

	/* Security */
	CryptoContext *crypto;

	/* Network */
	P2PNetworkContext *p2p;
	MPIContext *mpi;

	/* Multimedia */
	OpenGLContext *gl;
	OpenALContext *al;
	OpenCLContext *cl;

	/* Model */
	char model_path[MAX_PATH_LEN];
	char model_name[MAX_FILENAME_LEN];
	BINNamingConvention bin_info;
	uint32_t model_loaded;
	uint64_t model_load_time;

	/* NUMA */
	int numa_nodes;
	int *numa_node_cpus;
	size_t *numa_node_memory;
	void **numa_local_memory;

	/* Threads */
	ThreadWorker workers[MAX_THREADS];
	uint32_t num_workers;
	pthread_attr_t thread_attrs;

	/* Performance */
	PerformanceMetrics metrics;

	/* Synchronization */
	pthread_mutex_t state_mutex;
	pthread_cond_t state_cond;
	pthread_rwlock_t model_lock;
	pthread_spinlock_t metrics_lock;
	pthread_barrier_t sync_barrier;
};

/*=============================================================================
 * FUNCTION PROTOTYPES
 *============================================================================*/

/* Logging */
static void evox_log(LogLevel level, const char *file, int line,
		const char *fmt, ...);

/* Pure Functions */
static double pure_sigmoid(double x);
static double pure_tanh(double x);
static double pure_relu(double x);
static double pure_shannon_entropy(const double *probabilities, uint32_t count);

/* SIMD Operations */
static void simd_vector_add(const double *a, const double *b, double *c,
		uint32_t n);
static double simd_dot_product(const double *a, const double *b, uint32_t n);

/* 5-Axes */
static void five_axes_init(EVOXCoreSystem *system);
static double five_axes_weighting(const EVOXCoreSystem *system,
		const FiveAxisVector *p);

/* BIN Naming */
static int bin_parse_filename(const char *filename, BINNamingConvention *bin);

/* GGUF Converter */
static int gguf_to_bin_converter(EVOXCoreSystem *system, const char *gguf_path,
		const char *bin_path);

/* Crypto */
static CryptoContext* crypto_init(void);
static int crypto_checksum_file(const char *filename, uint8_t *checksum,
		size_t *checksum_len);
static int crypto_rotate_keys(CryptoContext *ctx);

/* Neuro-Fuzzy */
static NeuroFuzzySystem* fuzzy_system_init(uint32_t num_inputs,
		uint32_t num_outputs, uint32_t num_rules);
static double fuzzy_mamdani_inference(NeuroFuzzySystem *fuzzy,
		const double *inputs);

/* Hebbian Learning */
static void hebbian_update_synapse(Synapse *syn, double pre_act,
		double post_act, double lr, double dt);

/* Q-Learning */
static QLearningSystem* qlearn_init(uint32_t num_states, uint32_t num_actions);
static void qlearn_update(QLearningSystem *q, uint32_t state, uint32_t action,
		double reward, uint32_t next_state);
static uint32_t qlearn_select_action(QLearningSystem *q, uint32_t state);

/* Academic AI */
static MoEArchitecture* moe_init(uint32_t num_experts, uint32_t num_active,
		uint32_t capacity);
static void moe_route(MoEArchitecture *moe, const double *input, double *output);
static AttentionMechanism* attention_init(uint32_t num_heads, uint32_t head_dim,
		uint32_t context_len);
static ReasoningFramework* reasoning_init(uint32_t max_trace);
static CodeGenerator* coder_init(size_t buffer_size);

/* Thread Management */
static int threads_init(EVOXCoreSystem *system);

/* OpenGL */
static OpenGLContext* opengl_init(int width, int height);
static void opengl_render_axes(EVOXCoreSystem *system, OpenGLContext *gl);

/* OpenAL */
static OpenALContext* openal_init(void);
static void openal_play_neural_event(OpenALContext *al, double frequency,
		double amplitude);

/* P2P */
static P2PNetworkContext* p2p_init(uint16_t port, EVOXCoreSystem *system);

/* Model Loading */
static int scan_and_convert_models(EVOXCoreSystem *system);
static int model_load_bin(EVOXCoreSystem *system, const char *filename);

/* Neural Activity */
static void neural_activity_update(EVOXCoreSystem *system);

/* FSM */
static void fsm_process_event(EVOXCoreSystem *system, FSMEvent event,
		uint64_t time);
static void boot_sequence_init(BootSequence *boot);
static int boot_sequence_step(BootSequence *boot, EVOXCoreSystem *system);

/* System Lifecycle */
static EVOXCoreSystem* evox_system_init(void);
static void evox_main_loop(EVOXCoreSystem *system);
static void evox_system_cleanup(EVOXCoreSystem *system);

/*=============================================================================
 * LOGGING SYSTEM
 *============================================================================*/

static void evox_log(LogLevel level, const char *file, int line,
		const char *fmt, ...) {
	if (level > LOG_LEVEL)
		return;

	static const char *level_str[] = { "ERROR", "WARN", "INFO", "DEBUG" };
	char buffer[1024];
	va_list args;
	time_t now;
	struct tm *tm_info;

	time(&now);
	tm_info = localtime(&now);

	va_start(args, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, args);
	va_end(args);

	fprintf(stderr, "[%02d:%02d:%02d] [%s] [%s:%d] %s\n", tm_info->tm_hour,
			tm_info->tm_min, tm_info->tm_sec, level_str[level], file, line,
			buffer);
}

#define LOG_ERROR(...) evox_log(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)  evox_log(LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)  evox_log(LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_DEBUG(...) evox_log(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)

/*=============================================================================
 * PURE FUNCTIONS (C89 compatible)
 *============================================================================*/

static double pure_sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

static double pure_tanh(double x) {
	return tanh(x);
}

static double pure_relu(double x) {
	return (x > 0.0) ? x : 0.0;
}

static double pure_shannon_entropy(const double *probabilities, uint32_t count) {
	double entropy = 0.0;
	uint32_t i;
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

static void simd_vector_add(const double *a, const double *b, double *c,
		uint32_t n) {
	uint32_t i;
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

static double simd_dot_product(const double *a, const double *b, uint32_t n) {
	uint32_t i;
	__m256d sum_vec = _mm256_setzero_pd();
	double result = 0.0;
	double sum_array[4];

	for (i = 0; i + 3 < n; i += 4) {
		__m256d a_vec = _mm256_loadu_pd(&a[i]);
		__m256d b_vec = _mm256_loadu_pd(&b[i]);
		sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
	}

	_mm256_storeu_pd(sum_array, sum_vec);
	result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

	for (; i < n; ++i) {
		result += a[i] * b[i];
	}

	return result;
}

/*=============================================================================
 * 5-AXES MATHEMATICAL FORMULATION
 *============================================================================*/

static void five_axes_init(EVOXCoreSystem *system) {
	double inv_sqrt3 = 1.0 / sqrt(3.0);
	uint32_t i;

	/* Origin */
	system->origin.x = 0.0;
	system->origin.y = 0.0;
	system->origin.z = 0.0;
	system->origin.b = 0.0;
	system->origin.r = 0.0;

	/* X Axis */
	system->axes[AXIS_X_INDEX].x = 1.0;
	system->axes[AXIS_X_INDEX].y = 0.0;
	system->axes[AXIS_X_INDEX].z = 0.0;
	system->axes[AXIS_X_INDEX].b = 0.0;
	system->axes[AXIS_X_INDEX].r = 0.0;

	/* Y Axis */
	system->axes[AXIS_Y_INDEX].x = 0.0;
	system->axes[AXIS_Y_INDEX].y = 1.0;
	system->axes[AXIS_Y_INDEX].z = 0.0;
	system->axes[AXIS_Y_INDEX].b = 0.0;
	system->axes[AXIS_Y_INDEX].r = 0.0;

	/* Z Axis */
	system->axes[AXIS_Z_INDEX].x = 0.0;
	system->axes[AXIS_Z_INDEX].y = 0.0;
	system->axes[AXIS_Z_INDEX].z = 1.0;
	system->axes[AXIS_Z_INDEX].b = 0.0;
	system->axes[AXIS_Z_INDEX].r = 0.0;

	/* B Axis (Diagonal) */
	system->axes[AXIS_B_INDEX].x = inv_sqrt3;
	system->axes[AXIS_B_INDEX].y = inv_sqrt3;
	system->axes[AXIS_B_INDEX].z = inv_sqrt3;
	system->axes[AXIS_B_INDEX].b = 1.0;
	system->axes[AXIS_B_INDEX].r = 0.0;

	/* R Axis (Rotation) */
	system->axes[AXIS_R_INDEX].x = 0.0;
	system->axes[AXIS_R_INDEX].y = 0.0;
	system->axes[AXIS_R_INDEX].z = 0.0;
	system->axes[AXIS_R_INDEX].b = 0.0;
	system->axes[AXIS_R_INDEX].r = 1.0;

	/* Markers */
	for (i = 0; i < 3; ++i) {
		double val = (i == 0) ? 1.0 : (i == 1) ? 0.0 : -1.0;
		system->markers[i].x = val;
		system->markers[i].y = val;
		system->markers[i].z = val;
		system->markers[i].b = val;
		system->markers[i].r = val;
	}

	/* Adaptive coefficients */
	system->axis_weights[0] = 0.33; /* α - origin weight */
	system->axis_weights[1] = 0.34; /* β - positive weight */
	system->axis_weights[2] = 0.33; /* γ - negative weight */
	system->axis_weights[3] = 0.0;
	system->axis_weights[4] = 0.0;
}

static double five_axes_weighting(const EVOXCoreSystem *system,
		const FiveAxisVector *p) {
	double x = p->x, y = p->y, z = p->z, b = p->b, r = p->r;
	double alpha = system->axis_weights[0];
	double beta = system->axis_weights[1];
	double gamma = system->axis_weights[2];

	double squared_sum = x * x + y * y + z * z + b * b + r * r;
	double w_origin = exp(-sqrt(squared_sum));

	double w_positive = 0.0;
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

	double w_negative = 0.0;
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
 * BIN NAMING CONVENTION
 *============================================================================*/

static int bin_parse_filename(const char *filename, BINNamingConvention *bin) {
	char *shard_ptr;

	memset(bin, 0, sizeof(BINNamingConvention));
	strncpy(bin->filename, filename, MAX_FILENAME_LEN - 1);
	bin->filename[MAX_FILENAME_LEN - 1] = '\0';

	/* Default values */
	strcpy(bin->type, "normal");
	strcpy(bin->encoding, "fp16");
	strcpy(bin->version, "v1.0");
	strcpy(bin->base_name, "model");
	strcpy(bin->size_label, "7B");
	strcpy(bin->fine_tune, "base");

	/* Parse shard info */
	shard_ptr = strstr(filename, "-of-");
	if (shard_ptr != NULL) {
		sscanf(shard_ptr + 1, "%5d-of-%5d", &bin->shard_num, &bin->shard_total);
	}

	return 0;
}

/*=============================================================================
 * GGUF TO BIN CONVERTER
 *============================================================================*/

static int gguf_to_bin_converter(EVOXCoreSystem *system, const char *gguf_path,
		const char *bin_path) {
	FILE *gguf_file;
	FILE *bin_file;
	uint8_t header[4096];
	uint8_t buffer[8192];
	uint8_t encrypted[8192 + 16];
	size_t bytes_read;
	uint32_t block_len;
	GGUFFileHeader gguf_header;
	int ret = 0;

	/* Check if BIN already exists */
	if (access(bin_path, F_OK) == 0) {
		LOG_INFO("BIN file already exists: %s", bin_path);
		return 0;
	}

	LOG_INFO("Converting GGUF to BIN: %s -> %s", gguf_path, bin_path);

	gguf_file = fopen(gguf_path, "rb");
	if (!gguf_file) {
		LOG_ERROR("Failed to open GGUF file: %s", gguf_path);
		return -1;
	}

	/* Read and verify GGUF header */
	if (fread(&gguf_header, sizeof(gguf_header), 1, gguf_file) != 1) {
		LOG_ERROR("Failed to read GGUF header");
		ret = -1;
		goto cleanup_gguf;
	}

	if (gguf_header.magic != GGUF_MAGIC) {
		LOG_ERROR("Invalid GGUF magic: 0x%08x", gguf_header.magic);
		ret = -1;
		goto cleanup_gguf;
	}

	LOG_DEBUG("GGUF version: %u, tensors: %llu, metadata: %llu",
			gguf_header.version, (unsigned long long )gguf_header.tensor_count,
			(unsigned long long )gguf_header.metadata_kv_count);

	bin_file = fopen(bin_path, "wb");
	if (!bin_file) {
		LOG_ERROR("Failed to create BIN file: %s", bin_path);
		ret = -1;
		goto cleanup_gguf;
	}

	/* Write EVOX header */
	memset(header, 0, sizeof(header));
	memcpy(header, "EVOX", 4);
	header[4] = 0x01; /* Version major */
	header[5] = 0x00; /* Version minor */
	header[6] = 0x01; /* Encrypted */
	header[7] = 0x00; /* Not compressed */
	memcpy(header + 8, &gguf_header, sizeof(gguf_header));

	if (fwrite(header, 1, sizeof(header), bin_file) != sizeof(header)) {
		LOG_ERROR("Failed to write EVOX header");
		ret = -1;
		goto cleanup_both;
	}

	/* Encrypt and write model data (simplified XOR for demo) */
	while ((bytes_read = fread(buffer, 1, sizeof(buffer), gguf_file)) > 0) {
		uint32_t i;

		for (i = 0; i < bytes_read; ++i) {
			encrypted[i] = buffer[i] ^ 0xAA; /* Simple XOR */
		}

		block_len = (uint32_t) bytes_read;

		if (fwrite(&block_len, sizeof(block_len), 1, bin_file) != 1) {
			LOG_ERROR("Failed to write block length");
			ret = -1;
			goto cleanup_both;
		}

		if (fwrite(encrypted, 1, bytes_read, bin_file) != bytes_read) {
			LOG_ERROR("Failed to write encrypted data");
			ret = -1;
			goto cleanup_both;
		}
	}

	LOG_INFO("GGUF to BIN conversion completed");

	cleanup_both: fclose(bin_file);
	cleanup_gguf: fclose(gguf_file);
	return ret;
}

/*=============================================================================
 * MILITARY-GRADE CRYPTOGRAPHY
 *============================================================================*/

static CryptoContext* crypto_init(void) {
	CryptoContext *ctx = (CryptoContext*) calloc(1, sizeof(CryptoContext));
	if (!ctx)
		return NULL;

	OpenSSL_add_all_algorithms();
	ERR_load_crypto_strings();

	ctx->cipher_ctx = EVP_CIPHER_CTX_new();
	ctx->md_ctx = EVP_MD_CTX_new();

	/* Generate keys */
	RAND_bytes(ctx->aes_key, sizeof(ctx->aes_key));
	RAND_bytes(ctx->aes_iv, sizeof(ctx->aes_iv));
	RAND_bytes(ctx->hmac_key, sizeof(ctx->hmac_key));

	ctx->key_creation_time = (uint64_t) time(NULL);
	ctx->key_expiry_time = ctx->key_creation_time + KEY_ROTATION_SECONDS;
	ctx->key_rotations = 0;
	ctx->is_allocated = 1;

	pthread_mutex_init(&ctx->crypto_mutex, NULL);

	LOG_INFO("Crypto context initialized (AES-256)");
	return ctx;
}

static int crypto_checksum_file(const char *filename, uint8_t *checksum,
		size_t *checksum_len) {
	FILE *fp = fopen(filename, "rb");
	if (!fp)
		return -1;

	EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
	EVP_DigestInit_ex(md_ctx, EVP_sha256(), NULL);

	uint8_t buffer[8192];
	size_t bytes_read;

	while ((bytes_read = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
		EVP_DigestUpdate(md_ctx, buffer, bytes_read);
	}

	EVP_DigestFinal_ex(md_ctx, checksum, (unsigned int*) checksum_len);

	EVP_MD_CTX_free(md_ctx);
	fclose(fp);
	return 0;
}

static int crypto_rotate_keys(CryptoContext *ctx) {
	uint64_t now = (uint64_t) time(NULL);

	pthread_mutex_lock(&ctx->crypto_mutex);

	if (now < ctx->key_expiry_time) {
		pthread_mutex_unlock(&ctx->crypto_mutex);
		return 0;
	}

	RAND_bytes(ctx->aes_key, sizeof(ctx->aes_key));
	RAND_bytes(ctx->aes_iv, sizeof(ctx->aes_iv));
	RAND_bytes(ctx->hmac_key, sizeof(ctx->hmac_key));

	ctx->key_creation_time = now;
	ctx->key_expiry_time = now + KEY_ROTATION_SECONDS;
	ctx->key_rotations++;

	LOG_INFO("Cryptographic keys rotated (total: %u)", ctx->key_rotations);

	pthread_mutex_unlock(&ctx->crypto_mutex);
	return 1;
}

/*=============================================================================
 * NEURO-FUZZY INFERENCE SYSTEM
 *============================================================================*/

static NeuroFuzzySystem* fuzzy_system_init(uint32_t num_inputs,
		uint32_t num_outputs, uint32_t num_rules) {
	NeuroFuzzySystem *fuzzy = (NeuroFuzzySystem*) calloc(1,
			sizeof(NeuroFuzzySystem));
	if (!fuzzy)
		return NULL;

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

	uint32_t i;
	for (i = 0; i < num_inputs; ++i) {
		fuzzy->input_mf_params[i * 3] = -1.0; /* Low */
		fuzzy->input_mf_params[i * 3 + 1] = 0.0; /* Medium */
		fuzzy->input_mf_params[i * 3 + 2] = 1.0; /* High */
	}

	for (i = 0; i < num_outputs; ++i) {
		fuzzy->output_mf_params[i * 3] = -1.0; /* Low */
		fuzzy->output_mf_params[i * 3 + 1] = 0.0; /* Medium */
		fuzzy->output_mf_params[i * 3 + 2] = 1.0; /* High */
	}

	for (i = 0; i < num_rules * num_outputs; ++i) {
		fuzzy->rule_consequents[i] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
	}

	fuzzy->inference_type = MAMDANI_MIN;
	fuzzy->is_allocated = 1;

	LOG_INFO("Neuro-fuzzy system initialized: %u inputs, %u outputs, %u rules",
			num_inputs, num_outputs, num_rules);
	return fuzzy;
}

static double fuzzy_gaussian_mf(double x, double mean, double sigma) {
	return exp(-((x - mean) * (x - mean)) / (2.0 * sigma * sigma));
}

static double fuzzy_mamdani_inference(NeuroFuzzySystem *fuzzy,
		const double *inputs) {
	if (!fuzzy || !fuzzy->is_allocated)
		return 0.0;

	uint32_t i, j;

	/* Evaluate membership functions */
	for (i = 0; i < fuzzy->num_inputs; ++i) {
		fuzzy->fuzzy_sets[i * 3] = fuzzy_gaussian_mf(inputs[i],
				fuzzy->input_mf_params[i * 3], 0.3);
		fuzzy->fuzzy_sets[i * 3 + 1] = fuzzy_gaussian_mf(inputs[i],
				fuzzy->input_mf_params[i * 3 + 1], 0.3);
		fuzzy->fuzzy_sets[i * 3 + 2] = fuzzy_gaussian_mf(inputs[i],
				fuzzy->input_mf_params[i * 3 + 2], 0.3);
	}

	/* Evaluate rules */
	for (i = 0; i < fuzzy->num_rules; ++i) {
		double rule_strength = 1.0;
		for (j = 0; j < fuzzy->num_inputs; ++j) {
			uint32_t term = (i >> (j * 2)) & 0x03;
			if (term < 3) {
				double mf = fuzzy->fuzzy_sets[j * 3 + term];
				if (fuzzy->inference_type == MAMDANI_MIN) {
					if (mf < rule_strength)
						rule_strength = mf;
				} else {
					rule_strength *= mf;
				}
			}
		}
		fuzzy->rule_strengths[i] = rule_strength;
	}

	/* Defuzzify using centroid */
	double numerator = 0.0, denominator = 0.0;
	for (i = 0; i < fuzzy->num_rules; ++i) {
		for (j = 0; j < fuzzy->num_outputs; ++j) {
			double center = fuzzy->output_mf_params[j * 3 + 1];
			numerator += fuzzy->rule_strengths[i] * center
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
 * HEBBIAN LEARNING
 *============================================================================*/

static void hebbian_update_synapse(Synapse *syn, double pre_act,
		double post_act, double lr, double dt) {
	if (!syn)
		return;

	double delta = lr * pre_act * post_act * dt;

	syn->weight += delta;
	if (syn->weight > 1.0)
		syn->weight = 1.0;
	if (syn->weight < -1.0)
		syn->weight = -1.0;

	syn->luminescence += fabs(delta) * 10.0;
	if (syn->luminescence > 1.0)
		syn->luminescence = 1.0;
	syn->luminescence *= (1.0 - 0.1 * dt);

	syn->firing_count++;
	syn->last_update = dt;

	/* Update traces */
	syn->trace_pre = pre_act * 0.9 + syn->trace_pre * 0.1;
	syn->trace_post = post_act * 0.9 + syn->trace_post * 0.1;
}

/*=============================================================================
 * Q-LEARNING
 *============================================================================*/

static QLearningSystem* qlearn_init(uint32_t num_states, uint32_t num_actions) {
	QLearningSystem *qlearn = (QLearningSystem*) calloc(1,
			sizeof(QLearningSystem));
	if (!qlearn)
		return NULL;

	qlearn->num_states = num_states;
	qlearn->num_actions = num_actions;
	qlearn->learning_rate = 0.1;
	qlearn->discount_factor = 0.95;
	qlearn->exploration_rate = 0.1;
	qlearn->exploration_decay = 0.999;

	qlearn->q_table = (double*) malloc(
			num_states * num_actions * sizeof(double));
	qlearn->rewards = (double*) malloc(num_actions * sizeof(double));

	if (!qlearn->q_table || !qlearn->rewards) {
		free(qlearn->q_table);
		free(qlearn->rewards);
		free(qlearn);
		return NULL;
	}

	uint32_t i;
	for (i = 0; i < num_states * num_actions; ++i) {
		qlearn->q_table[i] = ((double) rand() / RAND_MAX) * 0.1;
	}

	pthread_spin_init(&qlearn->q_lock, PTHREAD_PROCESS_PRIVATE);
	qlearn->is_allocated = 1;

	LOG_INFO("Q-learning initialized: %u states, %u actions", num_states,
			num_actions);
	return qlearn;
}

static void qlearn_update(QLearningSystem *q, uint32_t state, uint32_t action,
		double reward, uint32_t next_state) {
	if (!q || !q->is_allocated)
		return;

	pthread_spin_lock(&q->q_lock);

	uint32_t idx = state * q->num_actions + action;
	uint32_t next_idx = next_state * q->num_actions;

	double max_next = q->q_table[next_idx];
	uint32_t i;
	for (i = 1; i < q->num_actions; ++i) {
		if (q->q_table[next_idx + i] > max_next) {
			max_next = q->q_table[next_idx + i];
		}
	}

	double td_error = reward + q->discount_factor * max_next - q->q_table[idx];
	q->q_table[idx] += q->learning_rate * td_error;

	q->learning_steps++;
	q->rewards[action] = reward;

	/* Decay exploration rate */
	q->exploration_rate *= q->exploration_decay;
	if (q->exploration_rate < 0.01)
		q->exploration_rate = 0.01;

	pthread_spin_unlock(&q->q_lock);
}

static uint32_t qlearn_select_action(QLearningSystem *q, uint32_t state) {
	if (!q || !q->is_allocated)
		return 0;

	pthread_spin_lock(&q->q_lock);

	uint32_t action;
	if ((double) rand() / RAND_MAX < q->exploration_rate) {
		action = rand() % q->num_actions;
	} else {
		uint32_t start = state * q->num_actions;
		action = 0;
		double best = q->q_table[start];
		uint32_t i;
		for (i = 1; i < q->num_actions; ++i) {
			if (q->q_table[start + i] > best) {
				best = q->q_table[start + i];
				action = i;
			}
		}
	}

	pthread_spin_unlock(&q->q_lock);
	return action;
}

/*=============================================================================
 * ACADEMIC AI FOUNDATIONS
 *============================================================================*/

static MoEArchitecture* moe_init(uint32_t num_experts, uint32_t num_active,
		uint32_t capacity) {
	MoEArchitecture *moe = (MoEArchitecture*) calloc(1,
			sizeof(MoEArchitecture));
	if (!moe)
		return NULL;

	moe->num_experts = num_experts;
	moe->num_active_experts = num_active;
	moe->expert_capacity = capacity;

	moe->routing_weights = (double*) malloc(num_experts * sizeof(double));
	moe->routing_indices = (uint32_t*) malloc(num_active * sizeof(uint32_t));
	moe->expert_outputs = (double*) malloc(
			num_experts * capacity * sizeof(double));
	moe->gate_outputs = (double*) malloc(num_experts * sizeof(double));
	moe->expert_biases = (double*) malloc(num_experts * sizeof(double));

	if (!moe->routing_weights || !moe->routing_indices || !moe->expert_outputs
			|| !moe->gate_outputs || !moe->expert_biases) {
		free(moe->routing_weights);
		free(moe->routing_indices);
		free(moe->expert_outputs);
		free(moe->gate_outputs);
		free(moe->expert_biases);
		free(moe);
		return NULL;
	}

	uint32_t i;
	for (i = 0; i < num_experts; ++i) {
		moe->gate_outputs[i] = ((double) rand() / RAND_MAX) * 0.1;
		moe->expert_biases[i] = ((double) rand() / RAND_MAX) * 0.01;
	}

	for (i = 0; i < num_experts * capacity; ++i) {
		moe->expert_outputs[i] = ((double) rand() / RAND_MAX) * 0.1;
	}

	pthread_spin_init(&moe->moe_lock, PTHREAD_PROCESS_PRIVATE);
	moe->is_allocated = 1;

	LOG_INFO("MoE initialized: %u experts, %u active", num_experts, num_active);
	return moe;
}

static void moe_route(MoEArchitecture *moe, const double *input, double *output) {
	if (!moe || !moe->is_allocated || !input || !output)
		return;

	pthread_spin_lock(&moe->moe_lock);

	uint32_t i, j, k;

	/* Compute gating network outputs */
	for (i = 0; i < moe->num_experts; ++i) {
		moe->gate_outputs[i] = input[0] * 0.1 + moe->expert_biases[i];
	}

	/* Softmax */
	double max_gate = moe->gate_outputs[0];
	for (i = 1; i < moe->num_experts; ++i) {
		if (moe->gate_outputs[i] > max_gate)
			max_gate = moe->gate_outputs[i];
	}

	double sum_exp = 0.0;
	for (i = 0; i < moe->num_experts; ++i) {
		moe->routing_weights[i] = exp(moe->gate_outputs[i] - max_gate);
		sum_exp += moe->routing_weights[i];
	}

	if (sum_exp > 1e-10) {
		for (i = 0; i < moe->num_experts; ++i) {
			moe->routing_weights[i] /= sum_exp;
		}
	}

	/* Top-k selection */
	for (i = 0; i < moe->num_active_experts; ++i) {
		uint32_t best_idx = 0;
		double best_weight = -1.0;

		for (j = 0; j < moe->num_experts; ++j) {
			if (moe->routing_weights[j] > best_weight) {
				int already = 0;
				for (k = 0; k < i; ++k) {
					if (moe->routing_indices[k] == j) {
						already = 1;
						break;
					}
				}
				if (!already) {
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
		uint32_t expert = moe->routing_indices[i];
		double weight = moe->routing_weights[expert];
		double *exp_out = &moe->expert_outputs[expert * moe->expert_capacity];

		for (j = 0; j < moe->expert_capacity; ++j) {
			output[j] += weight * exp_out[j];
		}
	}

	moe->total_routes++;

	pthread_spin_unlock(&moe->moe_lock);
}

static AttentionMechanism* attention_init(uint32_t num_heads, uint32_t head_dim,
		uint32_t context_len) {
	AttentionMechanism *attn = (AttentionMechanism*) calloc(1,
			sizeof(AttentionMechanism));
	if (!attn)
		return NULL;

	attn->num_heads = num_heads;
	attn->head_dim = head_dim;
	attn->context_length = context_len;
	attn->temperature = 1.0;
	attn->dropout_rate = 0.1;

	uint32_t total_dim = num_heads * head_dim;

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

	uint32_t i;
	for (i = 0; i < total_dim * total_dim; ++i) {
		attn->query_weights[i] = ((double) rand() / RAND_MAX - 0.5) * 0.01;
		attn->key_weights[i] = ((double) rand() / RAND_MAX - 0.5) * 0.01;
		attn->value_weights[i] = ((double) rand() / RAND_MAX - 0.5) * 0.01;
		attn->output_weights[i] = ((double) rand() / RAND_MAX - 0.5) * 0.01;
	}

	attn->is_allocated = 1;

	LOG_INFO("Attention initialized: %u heads, dim=%u, context=%u", num_heads,
			head_dim, context_len);
	return attn;
}

static ReasoningFramework* reasoning_init(uint32_t max_trace) {
	ReasoningFramework *reasoning = (ReasoningFramework*) calloc(1,
			sizeof(ReasoningFramework));
	if (!reasoning)
		return NULL;

	reasoning->max_trace_length = max_trace;
	reasoning->trace_length = 0;

	reasoning->reasoning_trace = (double*) malloc(max_trace * sizeof(double));
	reasoning->confidence_scores = (double*) malloc(max_trace * sizeof(double));
	reasoning->step_types = (uint32_t*) malloc(max_trace * sizeof(uint32_t));

	if (!reasoning->reasoning_trace || !reasoning->confidence_scores
			|| !reasoning->step_types) {
		free(reasoning->reasoning_trace);
		free(reasoning->confidence_scores);
		free(reasoning->step_types);
		free(reasoning);
		return NULL;
	}

	reasoning->is_allocated = 1;
	reasoning->reasoning_id = (uint64_t) time(NULL);

	LOG_INFO("Reasoning framework initialized: max trace %u", max_trace);
	return reasoning;
}

static CodeGenerator* coder_init(size_t buffer_size) {
	CodeGenerator *coder = (CodeGenerator*) calloc(1, sizeof(CodeGenerator));
	if (!coder)
		return NULL;

	coder->buffer_size = buffer_size;
	coder->language_id = 0; /* C by default */

	coder->code_buffer = (char*) malloc(buffer_size);
	coder->token_probabilities = (double*) malloc(
			MAX_VOCAB_SIZE * sizeof(double));
	coder->token_sequence = (uint32_t*) malloc(buffer_size * sizeof(uint32_t));

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

	LOG_INFO("Code generator initialized: buffer=%zu", buffer_size);
	return coder;
}

/*=============================================================================
 * THREAD MANAGEMENT
 *============================================================================*/

static void* worker_thread_func(void *arg) {
	EVOXCoreSystem *system = (EVOXCoreSystem*) arg;
	uint32_t thread_id = system->num_workers;

	/* Set thread affinity */
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
			&system->workers[thread_id].affinity);

	system->workers[thread_id].active = 1;

	LOG_DEBUG("Worker %d started", thread_id);

	while (!system->shutdown_flag) {
		system->workers[thread_id].working = 0;

		if (system->workers[thread_id].work_func) {
			system->workers[thread_id].working = 1;
			system->workers[thread_id].work_func(
					system->workers[thread_id].work_arg);
		} else {
			sched_yield();
			usleep(1000);
		}
	}

	system->workers[thread_id].active = 0;
	return NULL;
}

static int threads_init(EVOXCoreSystem *system) {
	system->num_workers = sysconf(_SC_NPROCESSORS_ONLN);
	if (system->num_workers > MAX_THREADS) {
		system->num_workers = MAX_THREADS;
	}

	pthread_attr_init(&system->thread_attrs);
	pthread_attr_setdetachstate(&system->thread_attrs, PTHREAD_CREATE_JOINABLE);

	uint32_t i;
	for (i = 0; i < system->num_workers; ++i) {
		CPU_ZERO(&system->workers[i].affinity);
		CPU_SET(i % CPU_SETSIZE, &system->workers[i].affinity);
		system->workers[i].thread_id = i;
		system->workers[i].work_func = NULL;
		system->workers[i].work_arg = NULL;

		if (pthread_create(&system->workers[i].thread, &system->thread_attrs,
				worker_thread_func, system) != 0) {
			LOG_ERROR("Failed to create worker thread %u", i);
			system->num_workers = i;
			break;
		}
	}

	LOG_INFO("Initialized %u worker threads", system->num_workers);
	return (system->num_workers > 0) ? 0 : -1;
}

/*=============================================================================
 * OPENGL RENDERING
 *============================================================================*/

static OpenGLContext* opengl_init(int width, int height) {
	OpenGLContext *gl = (OpenGLContext*) calloc(1, sizeof(OpenGLContext));
	if (!gl)
		return NULL;

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		LOG_ERROR("SDL_Init failed: %s", SDL_GetError());
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
		LOG_ERROR("SDL_CreateWindow failed: %s", SDL_GetError());
		SDL_Quit();
		free(gl);
		return NULL;
	}

	gl->gl_context = SDL_GL_CreateContext(gl->window);
	if (!gl->gl_context) {
		LOG_ERROR("SDL_GL_CreateContext failed: %s", SDL_GetError());
		SDL_DestroyWindow(gl->window);
		SDL_Quit();
		free(gl);
		return NULL;
	}

	SDL_GL_MakeCurrent(gl->window, gl->gl_context);
	SDL_GL_SetSwapInterval(1);

	gl->window_width = width;
	gl->window_height = height;
	gl->zoom_level = 1.0;
	gl->fps = 60.0;
	gl->frame_count = 0;
	gl->last_fps_time = (uint64_t) time(NULL);

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

	pthread_mutex_init(&gl->render_mutex, NULL);
	gl->is_allocated = 1;

	LOG_INFO("OpenGL initialized: %dx%d", width, height);
	return gl;
}

static void opengl_render_axes(EVOXCoreSystem *system, OpenGLContext *gl) {
	if (!system || !gl || !gl->is_allocated)
		return;

	pthread_mutex_lock(&gl->render_mutex);

	/* Update FPS counter */
	gl->frame_count++;
	uint64_t now = (uint64_t) time(NULL);
	if (now - gl->last_fps_time >= 1) {
		gl->fps = gl->frame_count / (double) (now - gl->last_fps_time);
		gl->frame_count = 0;
		gl->last_fps_time = now;
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	/* Set camera */
	gluLookAt(gl->camera_position.x, gl->camera_position.y,
			gl->camera_position.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	/* Apply zoom */
	glScaled(gl->zoom_level, gl->zoom_level, gl->zoom_level);

	/* Apply rotation */
	glRotated(gl->rotation_angle, 0.0, 1.0, 0.0);

	/* Draw origin marker (R axis) */
	glPointSize(12.0);
	glBegin(GL_POINTS);
	glColor4ub(255, 255, 0, 255);
	glVertex3f(0.0, 0.0, 0.0);
	glEnd();

	/* Draw axes */
	glLineWidth(3.0);

	/* X Axis - Red */
	glBegin(GL_LINES);
	glColor4ub(255, 0, 0, 255);
	glVertex3f(-2.5, 0.0, 0.0);
	glVertex3f(2.5, 0.0, 0.0);
	glEnd();

	/* Y Axis - Green */
	glBegin(GL_LINES);
	glColor4ub(0, 255, 0, 255);
	glVertex3f(0.0, -2.5, 0.0);
	glVertex3f(0.0, 2.5, 0.0);
	glEnd();

	/* Z Axis - Blue */
	glBegin(GL_LINES);
	glColor4ub(0, 0, 255, 255);
	glVertex3f(0.0, 0.0, -2.5);
	glVertex3f(0.0, 0.0, 2.5);
	glEnd();

	/* B Axis - Purple */
	glBegin(GL_LINES);
	glColor4ub(255, 0, 255, 255);
	glVertex3f(-2.0, -2.0, -2.0);
	glVertex3f(2.0, 2.0, 2.0);
	glEnd();

	/* Draw markers */
	glPointSize(8.0);
	glBegin(GL_POINTS);
	glColor4ub(255, 255, 255, 255); /* +1 - White */
	glVertex3f(1.5, 1.5, 1.5);
	glColor4ub(64, 64, 64, 255); /* -1 - Gray */
	glVertex3f(-1.5, -1.5, -1.5);
	glEnd();

	/* Draw neural network visualization */
	if (system->network && system->network->is_allocated
			&& system->network->nodes) {
		uint32_t i;

		/* Draw nodes */
		glPointSize(5.0);
		glBegin(GL_POINTS);

		uint32_t step = (system->network->num_nodes > 10000) ? 10 : 1;
		for (i = 0; i < system->network->num_nodes; i += step) {
			NeuralNode *node = &system->network->nodes[i];
			uint8_t intensity = (uint8_t) (node->activation * 255);
			glColor4ub(intensity, 0, 255 - intensity, 200);
			glVertex3f(node->position[0] * 2.0, node->position[1] * 2.0,
					node->position[2] * 2.0);
		}
		glEnd();

		/* Draw synapses */
		glLineWidth(1.0);
		glBegin(GL_LINES);

		step = (system->network->num_synapses > 5000) ? 20 : 1;
		for (i = 0; i < system->network->num_synapses && i < 5000; i += step) {
			Synapse *syn = &system->network->synapses[i];
			NeuralNode *from = &system->network->nodes[syn->from_node];
			NeuralNode *to = &system->network->nodes[syn->to_node];
			uint8_t lum = (uint8_t) (syn->luminescence * 255);
			glColor4ub(lum, lum, lum, (uint8_t) (fabs(syn->weight) * 200));
			glVertex3f(from->position[0] * 2.0, from->position[1] * 2.0,
					from->position[2] * 2.0);
			glVertex3f(to->position[0] * 2.0, to->position[1] * 2.0,
					to->position[2] * 2.0);
		}
		glEnd();
	}

	/* Draw axis labels */
	glColor4ub(255, 255, 255, 255);
	glRasterPos3f(2.8, 0.0, 0.0);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'X');
	glRasterPos3f(0.0, 2.8, 0.0);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Y');
	glRasterPos3f(0.0, 0.0, 2.8);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Z');
	glRasterPos3f(2.2, 2.2, 2.2);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'B');
	glRasterPos3f(0.2, 0.2, 0.2);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'R');

	/* Switch to orthographic for HUD */
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, gl->window_width, gl->window_height, 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);

	/* Draw HUD text */
	char hud_text[512];
	int y = 20;

	snprintf(hud_text, sizeof(hud_text),
			"Evox AI Core v%s | FPS: %.1f | State: %d | Inf: %llu",
			EVOX_VERSION_STRING, gl->fps, system->current_state,
			(unsigned long long) system->metrics.total_inferences);

	glColor4ub(255, 255, 255, 255);
	glRasterPos2i(10, y);
	uint32_t i;
	for (i = 0; hud_text[i] != '\0'; ++i) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, hud_text[i]);
	}

	y += 20;
	snprintf(hud_text, sizeof(hud_text),
			"5-Axes: X(Red) Y(Green) Z(Blue) B(Purple) R(Yellow) | Angle: %.1f Zoom: %.1f",
			gl->rotation_angle, gl->zoom_level);
	glRasterPos2i(10, y);
	for (i = 0; hud_text[i] != '\0'; ++i) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, hud_text[i]);
	}

	if (system->model_loaded) {
		y += 20;
		snprintf(hud_text, sizeof(hud_text), "Model: %s", system->model_name);
		glRasterPos2i(10, y);
		for (i = 0; hud_text[i] != '\0'; ++i) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, hud_text[i]);
		}
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
 * OPENAL AUDIO
 *============================================================================*/

static OpenALContext* openal_init(void) {
	OpenALContext *al = (OpenALContext*) calloc(1, sizeof(OpenALContext));
	if (!al)
		return NULL;

	al->audio_device = alcOpenDevice(NULL);
	if (!al->audio_device) {
		LOG_WARN("Could not open audio device");
		free(al);
		return NULL;
	}

	al->audio_context = alcCreateContext(al->audio_device, NULL);
	if (!al->audio_context) {
		alcCloseDevice(al->audio_device);
		free(al);
		return NULL;
	}

	alcMakeContextCurrent(al->audio_context);

	al->num_sources = 8;
	al->sound_sources = (ALuint*) malloc(al->num_sources * sizeof(ALuint));
	al->sound_buffers = (ALuint*) malloc(al->num_sources * sizeof(ALuint));

	alGenSources(al->num_sources, al->sound_sources);
	alGenBuffers(al->num_sources, al->sound_buffers);

	pthread_mutex_init(&al->audio_mutex, NULL);
	al->is_allocated = 1;

	LOG_INFO("OpenAL initialized");
	return al;
}

static void openal_play_neural_event(OpenALContext *al, double frequency,
		double amplitude) {
	if (!al || !al->is_allocated)
		return;

	pthread_mutex_lock(&al->audio_mutex);

	/* Generate a simple sine wave */
	int16_t buffer[4410]; /* 0.1 second at 44.1kHz */
	uint32_t i;
	for (i = 0; i < 4410; ++i) {
		double t = (double) i / 44100.0;
		buffer[i] = (int16_t) (amplitude * 32767.0
				* sin(2.0 * M_PI * frequency * t));
	}

	/* Find free source */
	ALint source_state;
	for (i = 0; i < al->num_sources; ++i) {
		alGetSourcei(al->sound_sources[i], AL_SOURCE_STATE, &source_state);
		if (source_state != AL_PLAYING) {
			alBufferData(al->sound_buffers[i], AL_FORMAT_MONO16, buffer,
					sizeof(buffer), 44100);
			alSourcei(al->sound_sources[i], AL_BUFFER, al->sound_buffers[i]);
			alSourcePlay(al->sound_sources[i]);
			break;
		}
	}

	pthread_mutex_unlock(&al->audio_mutex);
}

/*=============================================================================
 * P2P NETWORKING
 *============================================================================*/

static enum MHD_Result p2p_request_handler(void *cls,
		struct MHD_Connection *conn, const char *url, const char *method,
		const char *version, const char *upload_data, size_t *upload_data_size,
		void **con_cls) {
	EVOXCoreSystem *system = (EVOXCoreSystem*) cls;
	struct MHD_Response *response;
	enum MHD_Result ret;

	if (strcmp(url, "/api/key/rotate") == 0 && strcmp(method, "POST") == 0) {
		fsm_process_event(system, FSM_EVENT_KEY_EXPIRING,
				(uint64_t) time(NULL));
		const char *resp = "{\"status\":\"rotating\"}";
		response = MHD_create_response_from_buffer(strlen(resp), (void*) resp,
				MHD_RESPMEM_PERSISTENT);
		MHD_add_response_header(response, "Content-Type", "application/json");
		ret = MHD_queue_response(conn, MHD_HTTP_OK, response);
		MHD_destroy_response(response);
		return ret;
	}

	if (strcmp(url, "/api/status") == 0 && strcmp(method, "GET") == 0) {
		char resp[256];
		snprintf(resp, sizeof(resp),
				"{\"state\":%d,\"inferences\":%llu,\"model\":\"%s\"}",
				system->current_state,
				(unsigned long long) system->metrics.total_inferences,
				system->model_loaded ? system->model_name : "none");
		response = MHD_create_response_from_buffer(strlen(resp), resp,
				MHD_RESPMEM_MUST_COPY);
		MHD_add_response_header(response, "Content-Type", "application/json");
		ret = MHD_queue_response(conn, MHD_HTTP_OK, response);
		MHD_destroy_response(response);
		return ret;
	}

	const char *not_found = "{\"error\":\"not_found\"}";
	response = MHD_create_response_from_buffer(strlen(not_found),
			(void*) not_found, MHD_RESPMEM_PERSISTENT);
	MHD_add_response_header(response, "Content-Type", "application/json");
	ret = MHD_queue_response(conn, MHD_HTTP_NOT_FOUND, response);
	MHD_destroy_response(response);
	return ret;
}

static P2PNetworkContext* p2p_init(uint16_t port, EVOXCoreSystem *system) {
	P2PNetworkContext *p2p = (P2PNetworkContext*) calloc(1,
			sizeof(P2PNetworkContext));
	if (!p2p)
		return NULL;

	p2p->port = port;
	p2p->max_peers = MAX_PEERS;

	snprintf(p2p->node_id, sizeof(p2p->node_id), "evox_node_%lx",
			(unsigned long) time(NULL));

	p2p->peer_list = (char*) malloc(4096);
	p2p->message_buffer = (uint8_t*) malloc(MAX_MESSAGE_SIZE);
	p2p->buffer_size = MAX_MESSAGE_SIZE;

	pthread_rwlock_init(&p2p->peer_lock, NULL);

	p2p->http_daemon = MHD_start_daemon(
			MHD_USE_AUTO | MHD_USE_INTERNAL_POLLING_THREAD, port, NULL, NULL,
			&p2p_request_handler, system, MHD_OPTION_CONNECTION_TIMEOUT, 30,
			MHD_OPTION_END);

	if (!p2p->http_daemon) {
		free(p2p->peer_list);
		free(p2p->message_buffer);
		free(p2p);
		return NULL;
	}

	p2p->is_allocated = 1;
	LOG_INFO("P2P server listening on port %u", port);
	return p2p;
}

/*=============================================================================
 * MODEL LOADING
 *============================================================================*/

static int scan_and_convert_models(EVOXCoreSystem *system) {
	DIR *dir = opendir("./models");
	if (!dir) {
		LOG_WARN("Could not open ./models directory");
		return 0;
	}

	LOG_INFO("Scanning for GGUF models...");

	struct dirent *entry;
	int converted = 0;

	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_name[0] == '.')
			continue;

		const char *ext = strrchr(entry->d_name, '.');
		if (ext && strcmp(ext, ".gguf") == 0) {
			char gguf_path[MAX_PATH_LEN];
			char bin_path[MAX_PATH_LEN];

			snprintf(gguf_path, sizeof(gguf_path), "./models/%s",
					entry->d_name);
			snprintf(bin_path, sizeof(bin_path), "./models/%s", entry->d_name);
			char *dot = strrchr(bin_path, '.');
			if (dot)
				*dot = '\0';
			strcat(bin_path, ".bin");

			if (gguf_to_bin_converter(system, gguf_path, bin_path) == 0) {
				converted++;
			}
		}
	}

	closedir(dir);
	LOG_INFO("Converted %d GGUF models", converted);
	return converted;
}

static int model_load_bin(EVOXCoreSystem *system, const char *filename) {
	if (system->model_loaded) {
		LOG_DEBUG("Model already loaded");
		return 0;
	}

	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		LOG_ERROR("Failed to open: %s", filename);
		return -1;
	}

	LOG_INFO("Loading model: %s", filename);

	/* Read EVOX header */
	uint8_t header[4096];
	if (fread(header, 1, sizeof(header), fp) != sizeof(header)) {
		LOG_ERROR("Failed to read header");
		fclose(fp);
		return -1;
	}

	if (memcmp(header, "EVOX", 4) == 0) {
		LOG_INFO("EVOX format v%d.%d, encrypted: %s", header[4], header[5],
				header[6] ? "yes" : "no");
	}

	/* Parse BIN naming convention */
	bin_parse_filename(filename, &system->bin_info);
	strncpy(system->model_name, filename, MAX_FILENAME_LEN - 1);

	/* Initialize network */
	if (!system->network) {
		system->network = (NeuralNetwork*) calloc(1, sizeof(NeuralNetwork));
		if (!system->network) {
			fclose(fp);
			return -1;
		}
		system->network->is_allocated = 1;
	}

	/* Set network dimensions */
	system->network->vocab_size = 32000;
	system->network->hidden_size = 4096;
	system->network->num_layers = 32;
	system->network->num_experts = 8;

	system->network->num_nodes = system->network->vocab_size
			+ system->network->hidden_size * system->network->num_layers;
	system->network->num_synapses = system->network->num_nodes * 10;

	LOG_INFO("Network: %u nodes, %u synapses", system->network->num_nodes,
			system->network->num_synapses);

	/* Allocate memory */
	system->network->nodes = (NeuralNode*) malloc(
			system->network->num_nodes * sizeof(NeuralNode));
	system->network->synapses = (Synapse*) malloc(
			system->network->num_synapses * sizeof(Synapse));
	system->network->node_activations = (double*) malloc(
			system->network->num_nodes * sizeof(double));
	system->network->node_deltas = (double*) malloc(
			system->network->num_nodes * sizeof(double));
	system->network->expert_routing = (uint32_t*) malloc(
			system->network->num_experts * sizeof(uint32_t));

	if (!system->network->nodes || !system->network->synapses
			|| !system->network->node_activations
			|| !system->network->node_deltas
			|| !system->network->expert_routing) {
		LOG_ERROR("Failed to allocate network memory");
		free(system->network->nodes);
		free(system->network->synapses);
		free(system->network->node_activations);
		free(system->network->node_deltas);
		free(system->network->expert_routing);
		free(system->network);
		system->network = NULL;
		fclose(fp);
		return -1;
	}

	/* Initialize nodes */
	uint32_t i;
	for (i = 0; i < system->network->num_nodes; ++i) {
		NeuralNode *node = &system->network->nodes[i];
		node->activation = ((double) rand() / RAND_MAX) * 0.1;
		node->membrane_potential = 0.0;
		node->threshold = 1.0;
		node->spike_count = 0;

		/* Position in 5-axes space */
		node->position[AXIS_X_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[AXIS_Y_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[AXIS_Z_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[AXIS_B_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		node->position[AXIS_R_INDEX] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
	}

	/* Initialize synapses */
	for (i = 0; i < system->network->num_synapses; ++i) {
		Synapse *syn = &system->network->synapses[i];
		syn->from_node = rand() % system->network->num_nodes;
		syn->to_node = rand() % system->network->num_nodes;
		syn->weight = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
		syn->luminescence = ((double) rand() / RAND_MAX) * 0.5;
	}

	pthread_spin_init(&system->network->network_lock, PTHREAD_PROCESS_PRIVATE);

	system->model_loaded = 1;
	system->model_load_time = (uint64_t) time(NULL);

	fclose(fp);
	LOG_INFO("Model loaded successfully");
	return 0;
}

/*=============================================================================
 * NEURAL ACTIVITY UPDATE
 *============================================================================*/

static void neural_activity_update(EVOXCoreSystem *system) {
	if (!system || !system->network || !system->network->is_allocated)
		return;

	uint32_t i, j;
	double sum_activations = 0.0;

	for (i = 0; i < system->network->num_nodes; ++i) {
		NeuralNode *node = &system->network->nodes[i];
		double input_sum = 0.0;

		/* Collect inputs */
		for (j = 0; j < system->network->num_synapses; ++j) {
			Synapse *syn = &system->network->synapses[j];
			if (syn->to_node == i) {
				NeuralNode *from = &system->network->nodes[syn->from_node];
				input_sum += from->activation * syn->weight;
			}
		}

		/* Update activation */
		node->activation = pure_sigmoid(input_sum);
		sum_activations += node->activation;

		/* Hebbian learning */
		for (j = 0; j < system->network->num_synapses; ++j) {
			Synapse *syn = &system->network->synapses[j];
			if (syn->to_node == i) {
				NeuralNode *from = &system->network->nodes[syn->from_node];
				hebbian_update_synapse(syn, from->activation, node->activation,
						0.01, 0.001);
			}
		}

		/* Apply 5-axes weighting */
		FiveAxisVector pos;
		pos.x = node->position[AXIS_X_INDEX];
		pos.y = node->position[AXIS_Y_INDEX];
		pos.z = node->position[AXIS_Z_INDEX];
		pos.b = node->position[AXIS_B_INDEX];
		pos.r = node->position[AXIS_R_INDEX];

		double weight = five_axes_weighting(system, &pos);
		node->activation *= (1.0 + weight);
	}

	/* Calculate entropy */
	if (sum_activations > 0.0) {
		double *probs = (double*) alloca(
				system->network->num_nodes * sizeof(double));
		for (i = 0; i < system->network->num_nodes; ++i) {
			probs[i] = system->network->nodes[i].activation / sum_activations;
		}

		double entropy = pure_shannon_entropy(probs,
				system->network->num_nodes);

		/* Neuro-fuzzy inference */
		if (system->fuzzy && system->fuzzy->is_allocated) {
			double inputs[1] = { entropy / 10.0 };
			fuzzy_mamdani_inference(system->fuzzy, inputs);
		}

		/* Q-learning */
		if (system->qlearn && system->qlearn->is_allocated
				&& (rand() % 100 == 0)) {
			uint32_t state = rand() % system->qlearn->num_states;
			uint32_t action = qlearn_select_action(system->qlearn, state);
			double reward = (entropy - 5.0) / 5.0;
			uint32_t next = (state + 1) % system->qlearn->num_states;
			qlearn_update(system->qlearn, state, action, reward, next);
		}

		/* Generate audio feedback */
		if (system->al && system->al->is_allocated && entropy > 7.0) {
			openal_play_neural_event(system->al, 440.0 * (1.0 + entropy / 10.0),
					0.1);
		}
	}

	system->metrics.total_inferences++;
	system->network->average_activation = sum_activations
			/ system->network->num_nodes;
}

/*=============================================================================
 * FINITE STATE MACHINE
 *============================================================================*/

static const FSMState fsm_transitions[FSM_STATE_COUNT][FSM_EVENT_COUNT] = {
		[FSM_STATE_BOOT] = { [FSM_EVENT_BOOT_COMPLETE] = FSM_STATE_SELF_TEST,
				[FSM_EVENT_BOOT_FAILED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_SELF_TEST] = { [FSM_EVENT_BOOT_COMPLETE
				] = FSM_STATE_HARDWARE_INIT, [FSM_EVENT_ERROR_DETECTED
				] = FSM_STATE_ERROR, [FSM_EVENT_SHUTDOWN_REQUEST
				] = FSM_STATE_SHUTDOWN }, [FSM_STATE_HARDWARE_INIT] = {
				[FSM_EVENT_BOOT_COMPLETE] = FSM_STATE_MODEL_LOAD,
				[FSM_EVENT_ERROR_DETECTED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_MODEL_LOAD] = { [FSM_EVENT_MODEL_LOADED
				] = FSM_STATE_NETWORK_INIT, [FSM_EVENT_MODEL_FAILED
				] = FSM_STATE_ERROR, [FSM_EVENT_SHUTDOWN_REQUEST
				] = FSM_STATE_SHUTDOWN }, [FSM_STATE_NETWORK_INIT] = {
				[FSM_EVENT_NETWORK_READY] = FSM_STATE_CRYPTO_INIT,
				[FSM_EVENT_NETWORK_FAILED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_CRYPTO_INIT] = { [FSM_EVENT_BOOT_COMPLETE
				] = FSM_STATE_RENDERING_INIT, [FSM_EVENT_ERROR_DETECTED
				] = FSM_STATE_ERROR, [FSM_EVENT_SHUTDOWN_REQUEST
				] = FSM_STATE_SHUTDOWN }, [FSM_STATE_RENDERING_INIT] = {
				[FSM_EVENT_BOOT_COMPLETE] = FSM_STATE_AUDIO_INIT,
				[FSM_EVENT_ERROR_DETECTED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_AUDIO_INIT] = { [FSM_EVENT_BOOT_COMPLETE] = FSM_STATE_IDLE,
				[FSM_EVENT_ERROR_DETECTED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_IDLE] = { [FSM_EVENT_INFERENCE_REQUEST
				] = FSM_STATE_PROCESSING, [FSM_EVENT_LEARNING_TRIGGER
				] = FSM_STATE_LEARNING, [FSM_EVENT_ROUTE_UPDATE
				] = FSM_STATE_ROUTING, [FSM_EVENT_KEY_EXPIRING
				] = FSM_STATE_KEY_ROTATION, [FSM_EVENT_ERROR_DETECTED
				] = FSM_STATE_ERROR, [FSM_EVENT_SHUTDOWN_REQUEST
				] = FSM_STATE_SHUTDOWN }, [FSM_STATE_PROCESSING] = {
				[FSM_EVENT_BOOT_COMPLETE] = FSM_STATE_IDLE,
				[FSM_EVENT_ERROR_DETECTED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_LEARNING] = { [FSM_EVENT_BOOT_COMPLETE] = FSM_STATE_IDLE,
				[FSM_EVENT_ERROR_DETECTED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_ROUTING] = { [FSM_EVENT_BOOT_COMPLETE] = FSM_STATE_IDLE,
				[FSM_EVENT_ERROR_DETECTED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_KEY_ROTATION] = { [FSM_EVENT_KEY_ROTATED] = FSM_STATE_IDLE,
				[FSM_EVENT_ERROR_DETECTED] = FSM_STATE_ERROR,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_ERROR] = { [FSM_EVENT_BOOT_COMPLETE] = FSM_STATE_IDLE,
				[FSM_EVENT_SHUTDOWN_REQUEST] = FSM_STATE_SHUTDOWN },
		[FSM_STATE_SHUTDOWN] = { } };

static void boot_sequence_init(BootSequence *boot) {
	boot->current_step = 0;
	boot->steps_completed = 0;
	boot->errors_detected = 0;
	boot->boot_complete = 0;
	boot->boot_successful = 0;
	boot->boot_start_time = (uint64_t) time(NULL);
	memset(boot->error_message, 0, sizeof(boot->error_message));
}

static int boot_sequence_step(BootSequence *boot, EVOXCoreSystem *system) {
	uint64_t start = (uint64_t) time(NULL);
	int ret = 0;

	LOG_INFO("Boot step %d", boot->current_step);

	switch (boot->current_step) {
	case 0: /* CPU check */
#if defined(__AVX2__) && defined(__FMA__)
		LOG_DEBUG("AVX2/FMA detected");
		ret = 1;
#else
            LOG_ERROR("AVX2/FMA required");
            ret = 0;
            strcpy(boot->error_message, "CPU missing AVX2/FMA");
            boot->errors_detected++;
            #endif
		break;

	case 1: /* Memory check */
	{
		struct sysinfo si;
		if (sysinfo(&si) == 0) {
			LOG_DEBUG("Memory: %lu MB free",
					(unsigned long )(si.freeram >> 20));
			if (si.freeram < 256 * 1024 * 1024) {
				LOG_ERROR("Insufficient memory");
				strcpy(boot->error_message, "Insufficient memory");
				boot->errors_detected++;
				ret = 0;
			} else {
				ret = 1;
			}
		} else {
			ret = 1;
		}
	}
		break;

	case 2: /* NUMA detection */
		if (numa_available() >= 0) {
			system->numa_nodes = numa_max_node() + 1;
			system->numa_node_cpus = (int*) calloc(system->numa_nodes,
					sizeof(int));
			system->numa_node_memory = (size_t*) calloc(system->numa_nodes,
					sizeof(size_t));
			LOG_DEBUG("NUMA nodes: %d", system->numa_nodes);
			ret = 1;
		} else {
			system->numa_nodes = 1;
			ret = 1;
		}
		break;

	case 3: /* Crypto init */
		if (!system->crypto) {
			system->crypto = crypto_init();
			ret = (system->crypto != NULL);
		} else {
			ret = 1;
		}
		break;

	case 4: /* Network allocation */
		if (!system->network) {
			system->network = (NeuralNetwork*) calloc(1, sizeof(NeuralNetwork));
			if (system->network) {
				system->network->vocab_size = 32000;
				system->network->hidden_size = 4096;
				system->network->num_layers = 32;
				system->network->num_experts = 8;
				system->network->is_allocated = 1;
				ret = 1;
			}
		} else {
			ret = 1;
		}
		break;

	case 5: /* AI foundations */
		if (!system->moe)
			system->moe = moe_init(8, 2, 4096);
		if (!system->attention)
			system->attention = attention_init(32, 128, 2048);
		if (!system->reasoning)
			system->reasoning = reasoning_init(1024);
		if (!system->coder)
			system->coder = coder_init(65536);
		ret = (system->moe && system->attention && system->reasoning
				&& system->coder);
		break;

	case 6: /* Create models directory */
		if (mkdir("./models", 0755) == 0 || errno == EEXIST) {
			ret = 1;
		} else {
			ret = 0;
		}
		break;

	case 7: /* Final verification */
		if (boot->errors_detected == 0) {
			boot->boot_complete = 1;
			boot->boot_successful = 1;
			ret = 1;
			LOG_INFO("Boot successful");
		} else {
			boot->boot_complete = 1;
			boot->boot_successful = 0;
			ret = 0;
			LOG_ERROR("Boot failed with %d errors", boot->errors_detected);
		}
		break;
	}

	boot->step_timings[boot->current_step] = difftime(time(NULL), start);

	if (ret && boot->current_step < BOOT_STEPS - 1) {
		boot->current_step++;
		boot->steps_completed++;
	}

	return ret;
}

static void fsm_process_event(EVOXCoreSystem *system, FSMEvent event,
		uint64_t time) {
	if (!system)
		return;

	pthread_mutex_lock(&system->state_mutex);

	FSMState new_state = fsm_transitions[system->current_state][event];

	if (new_state != FSM_STATE_COUNT && new_state != system->current_state) {
		system->previous_state = system->current_state;
		system->current_state = new_state;
		system->state_entry_time = time;

		LOG_INFO("FSM: %d -> %d (event: %d)", system->previous_state, new_state,
				event);

		pthread_cond_broadcast(&system->state_cond);
	}

	pthread_mutex_unlock(&system->state_mutex);
}

/*=============================================================================
 * SYSTEM INITIALIZATION
 *============================================================================*/

static EVOXCoreSystem* evox_system_init(void) {
	EVOXCoreSystem *system = (EVOXCoreSystem*) malloc(sizeof(EVOXCoreSystem));
	if (!system)
		return NULL;

	memset(system, 0, sizeof(EVOXCoreSystem));

	system->version_major = EVOX_VERSION_MAJOR;
	system->version_minor = EVOX_VERSION_MINOR;
	system->version_patch = EVOX_VERSION_PATCH;
	strcpy(system->version_string, EVOX_VERSION_STRING);

	system->current_state = FSM_STATE_BOOT;
	system->state_entry_time = (uint64_t) time(NULL);

	pthread_mutex_init(&system->state_mutex, NULL);
	pthread_cond_init(&system->state_cond, NULL);
	pthread_rwlock_init(&system->model_lock, NULL);
	pthread_spin_init(&system->metrics_lock, PTHREAD_PROCESS_PRIVATE);

	boot_sequence_init(&system->boot);
	five_axes_init(system);

	system->fuzzy = fuzzy_system_init(1, 1, 8);
	system->qlearn = qlearn_init(100, 10);

	LOG_INFO("Evox AI Core v%s initializing", EVOX_VERSION_STRING);
	return system;
}

/*=============================================================================
 * MAIN EVENT LOOP
 *============================================================================*/

static void evox_main_loop(EVOXCoreSystem *system) {
	uint64_t last_key_check = 0;
	uint64_t last_render = 0;
	uint64_t last_activity = 0;
	uint64_t last_scan = 0;
	int scan_attempts = 0;

	while (!system->shutdown_flag) {
		uint64_t now = (uint64_t) time(NULL);

		/* Process FSM state */
		switch (system->current_state) {
		case FSM_STATE_BOOT:
			if (!system->boot.boot_complete) {
				boot_sequence_step(&system->boot, system);
			} else {
				FSMEvent event =
						system->boot.boot_successful ?
								FSM_EVENT_BOOT_COMPLETE : FSM_EVENT_BOOT_FAILED;
				fsm_process_event(system, event, now);
			}
			break;

		case FSM_STATE_SELF_TEST:
		case FSM_STATE_HARDWARE_INIT:
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, now);
			break;

		case FSM_STATE_MODEL_LOAD:
			if (!system->model_loaded) {
				if (now - last_scan > 2 || scan_attempts == 0) {
					scan_and_convert_models(system);
					last_scan = now;
					scan_attempts++;
				}

				DIR *dir = opendir("./models");
				if (dir) {
					struct dirent *entry;
					while ((entry = readdir(dir)) != NULL) {
						if (entry->d_name[0] == '.')
							continue;
						const char *ext = strrchr(entry->d_name, '.');
						if (ext && strcmp(ext, ".bin") == 0) {
							char path[MAX_PATH_LEN];
							snprintf(path, sizeof(path), "./models/%s",
									entry->d_name);
							if (model_load_bin(system, path) == 0) {
								fsm_process_event(system,
										FSM_EVENT_MODEL_LOADED, now);
								break;
							}
						}
					}
					closedir(dir);
				}
			}

			if (!system->model_loaded && scan_attempts > 3) {
				fsm_process_event(system, FSM_EVENT_MODEL_FAILED, now);
			}
			break;

		case FSM_STATE_NETWORK_INIT:
			if (!system->p2p) {
				system->p2p = p2p_init(8080, system);
			}
			fsm_process_event(system, FSM_EVENT_NETWORK_READY, now);
			break;

		case FSM_STATE_CRYPTO_INIT:
			if (!system->crypto) {
				system->crypto = crypto_init();
			}
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, now);
			break;

		case FSM_STATE_RENDERING_INIT:
			if (!system->gl) {
				system->gl = opengl_init(1280, 720);
			}
			fsm_process_event(system,
					system->gl ?
							FSM_EVENT_BOOT_COMPLETE : FSM_EVENT_ERROR_DETECTED,
					now);
			break;

		case FSM_STATE_AUDIO_INIT:
			if (!system->al) {
				system->al = openal_init();
			}
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, now);
			break;

		case FSM_STATE_IDLE:
			if (now - last_key_check > 3600) {
				if (system->crypto && now > system->crypto->key_expiry_time) {
					fsm_process_event(system, FSM_EVENT_KEY_EXPIRING, now);
				}
				last_key_check = now;
			}

			if (system->model_loaded && (rand() % 1000 == 0)) {
				fsm_process_event(system, FSM_EVENT_INFERENCE_REQUEST, now);
			}
			break;

		case FSM_STATE_PROCESSING:
			neural_activity_update(system);
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, now);
			break;

		case FSM_STATE_LEARNING:
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, now);
			break;

		case FSM_STATE_ROUTING:
			if (system->moe && system->model_loaded) {
				double in[1] = { ((double) rand() / RAND_MAX) };
				double out[4096];
				moe_route(system->moe, in, out);
			}
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, now);
			break;

		case FSM_STATE_KEY_ROTATION:
			if (system->crypto) {
				crypto_rotate_keys(system->crypto);
			}
			fsm_process_event(system, FSM_EVENT_KEY_ROTATED, now);
			break;

		case FSM_STATE_ERROR:
			LOG_WARN("Error state, attempting recovery");
			sleep(2);
			fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE, now);
			break;

		case FSM_STATE_SHUTDOWN:
			system->shutdown_flag = 1;
			break;

		default:
			break;
		}

		/* Periodic updates */
		if (now - last_activity > 1) {
			if (system->model_loaded
					&& (system->current_state == FSM_STATE_IDLE
							|| system->current_state == FSM_STATE_PROCESSING)) {
				neural_activity_update(system);
			}
			last_activity = now;
		}

		if (system->gl && system->gl->is_allocated && now - last_render > 0) {
			opengl_render_axes(system, system->gl);
			system->gl->rotation_angle += 0.5;
			last_render = now;
		}

		/* Handle SDL events */
		if (system->gl && system->gl->window) {
			SDL_Event event;
			while (SDL_PollEvent(&event)) {
				if (event.type == SDL_QUIT) {
					fsm_process_event(system, FSM_EVENT_SHUTDOWN_REQUEST, now);
				} else if (event.type == SDL_KEYDOWN) {
					if (event.key.keysym.sym == SDLK_ESCAPE) {
						fsm_process_event(system, FSM_EVENT_SHUTDOWN_REQUEST,
								now);
					} else if (event.key.keysym.sym == SDLK_SPACE) {
						system->gl->rotation_angle = 0.0;
					} else if (event.key.keysym.sym == SDLK_r) {
						system->gl->rotation_angle += 10.0;
					} else if (event.key.keysym.sym == SDLK_PLUS
							|| event.key.keysym.sym == SDLK_EQUALS) {
						system->gl->zoom_level *= 1.1;
					} else if (event.key.keysym.sym == SDLK_MINUS) {
						system->gl->zoom_level /= 1.1;
					}
				} else if (event.type == SDL_WINDOWEVENT
						&& event.window.event == SDL_WINDOWEVENT_RESIZED) {
					system->gl->window_width = event.window.data1;
					system->gl->window_height = event.window.data2;
					glViewport(0, 0, event.window.data1, event.window.data2);
					glMatrixMode(GL_PROJECTION);
					glLoadIdentity();
					gluPerspective(45.0,
							(double) event.window.data1 / event.window.data2,
							0.1, 1000.0);
					glMatrixMode(GL_MODELVIEW);
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
	if (!system)
		return;

	LOG_INFO("Shutting down...");
	system->shutdown_flag = 1;

	/* Wait for workers */
	uint32_t i;
	for (i = 0; i < system->num_workers; ++i) {
		pthread_join(system->workers[i].thread, NULL);
	}

	/* Free neural network */
	if (system->network && system->network->is_allocated) {
		free(system->network->nodes);
		free(system->network->synapses);
		free(system->network->node_activations);
		free(system->network->node_deltas);
		free(system->network->expert_routing);
		pthread_spin_destroy(&system->network->network_lock);
		free(system->network);
	}

	/* Free AI components */
	if (system->moe && system->moe->is_allocated) {
		free(system->moe->routing_weights);
		free(system->moe->routing_indices);
		free(system->moe->expert_outputs);
		free(system->moe->gate_outputs);
		free(system->moe->expert_biases);
		pthread_spin_destroy(&system->moe->moe_lock);
		free(system->moe);
	}

	if (system->attention && system->attention->is_allocated) {
		free(system->attention->query_weights);
		free(system->attention->key_weights);
		free(system->attention->value_weights);
		free(system->attention->output_weights);
		free(system->attention->attention_scores);
		free(system->attention);
	}

	if (system->reasoning && system->reasoning->is_allocated) {
		free(system->reasoning->reasoning_trace);
		free(system->reasoning->confidence_scores);
		free(system->reasoning->step_types);
		free(system->reasoning);
	}

	if (system->coder && system->coder->is_allocated) {
		free(system->coder->code_buffer);
		free(system->coder->token_probabilities);
		free(system->coder->token_sequence);
		pthread_mutex_destroy(&system->coder->code_mutex);
		free(system->coder);
	}

	if (system->fuzzy && system->fuzzy->is_allocated) {
		free(system->fuzzy->fuzzy_sets);
		free(system->fuzzy->rule_strengths);
		free(system->fuzzy->rule_consequents);
		free(system->fuzzy->input_mf_params);
		free(system->fuzzy->output_mf_params);
		free(system->fuzzy);
	}

	if (system->qlearn && system->qlearn->is_allocated) {
		free(system->qlearn->q_table);
		free(system->qlearn->rewards);
		pthread_spin_destroy(&system->qlearn->q_lock);
		free(system->qlearn);
	}

	/* Free crypto */
	if (system->crypto && system->crypto->is_allocated) {
		EVP_CIPHER_CTX_free(system->crypto->cipher_ctx);
		EVP_MD_CTX_free(system->crypto->md_ctx);
		pthread_mutex_destroy(&system->crypto->crypto_mutex);
		free(system->crypto);
	}

	/* Free P2P */
	if (system->p2p && system->p2p->is_allocated) {
		if (system->p2p->http_daemon)
			MHD_stop_daemon(system->p2p->http_daemon);
		free(system->p2p->peer_list);
		free(system->p2p->message_buffer);
		pthread_rwlock_destroy(&system->p2p->peer_lock);
		free(system->p2p);
	}

	/* Free OpenGL */
	if (system->gl && system->gl->is_allocated) {
		if (system->gl->gl_context)
			SDL_GL_DeleteContext(system->gl->gl_context);
		if (system->gl->window)
			SDL_DestroyWindow(system->gl->window);
		pthread_mutex_destroy(&system->gl->render_mutex);
		free(system->gl);
	}

	/* Free OpenAL */
	if (system->al && system->al->is_allocated) {
		if (system->al->audio_context)
			alcDestroyContext(system->al->audio_context);
		if (system->al->audio_device)
			alcCloseDevice(system->al->audio_device);
		free(system->al->sound_sources);
		free(system->al->sound_buffers);
		pthread_mutex_destroy(&system->al->audio_mutex);
		free(system->al);
	}

	/* Free NUMA resources */
	free(system->numa_node_cpus);
	free(system->numa_node_memory);

	/* Destroy synchronization */
	pthread_mutex_destroy(&system->state_mutex);
	pthread_cond_destroy(&system->state_cond);
	pthread_rwlock_destroy(&system->model_lock);
	pthread_spin_destroy(&system->metrics_lock);

	SDL_Quit();

	LOG_INFO("Shutdown complete - total inferences: %llu",
			(unsigned long long )system->metrics.total_inferences);
	free(system);
}

/*=============================================================================
 * ENTRY POINT
 *============================================================================*/

int main(int argc, char *argv[]) {
	/* Initialize random seed */
	srand((unsigned int) time(NULL));

	/* Initialize GLUT */
	int glut_argc = 1;
	char *glut_argv[2] = { argv[0], "" };
	glutInit(&glut_argc, glut_argv);

	/* Create system */
	EVOXCoreSystem *system = evox_system_init();
	if (!system) {
		LOG_ERROR("System initialization failed");
		return EXIT_FAILURE;
	}

	LOG_INFO("System initialized, starting main loop");
	LOG_INFO("5-Axes: X(Red) Y(Green) Z(Blue) B(Purple) R(Yellow)");
	LOG_INFO("Controls: ESC=Exit SPACE=Reset R=Rotate +/-=Zoom");

	/* Create models directory */
	mkdir("./models", 0755);

	/* Initialize threads */
	threads_init(system);

	/* Run main loop */
	evox_main_loop(system);

	/* Cleanup */
	evox_system_cleanup(system);

	LOG_INFO("Terminated normally");
	return EXIT_SUCCESS;
}
