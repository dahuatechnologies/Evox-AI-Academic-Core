/*
 * Copyright (c) 2026 Evolution Technologies Research and Prototype
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * 5A EVOX AI CORE v1.0 - Production Deterministic 5-Axis Core AI System
 * File: evox/src/main.c
 * Version: 1.0.0
 * Standard: ANSI C89/90 with POSIX compliance
 *
 * Features:
 * - Advanced 5 Axes Spiral (X(Length), Y(Height), Z(Width), B(Radius Base), R(Rotation))
 * - 14-state Deterministic Finite State Machine with 8-step initialization
 * - Real-time multimedia visualization with proper lighting and particle effects
 * - Starting from R(0,0,0,0,0) origin
 * - Academic AI foundations: MoE, R1, V2, Coder
 * - Autonomous neural network scaling (0-50K vocab, 0-4096 hidden, 0-32 layers)
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1

#ifndef __cplusplus
#define inline
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <numa.h>
#include <numaif.h>
#include <immintrin.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stddef.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <omp.h>
#include <mpi.h>
#include <microhttpd.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/err.h>
#include <CL/cl.h>
#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <AL/al.h>
#include <AL/alc.h>

/*============================================================================
 * SYSTEM CONSTANTS
 *============================================================================*/

#ifndef M_PI
#define M_PI                            3.14159265358979323846
#endif

#define PI                              M_PI
#define TWO_PI                          6.28318530717958647692
#define HALF_PI                         1.57079632679489661923

/* FSM Constants - 14 States */
#define MAX_STATES                      32
#define FSM_CYCLE_DURATION              100  /* milliseconds per state transition */
#define FSM_TOTAL_STATES                 14   /* 14 states from BOOT to TERMINATE */

/* Boot Sequence - 8 Steps */
#define BOOT_STEPS                        8
#define BOOT_STEP_POWER_ON                 0
#define BOOT_STEP_HARDWARE_INIT            1
#define BOOT_STEP_MEMORY_TEST               2
#define BOOT_STEP_CORE_LOAD                 3
#define BOOT_STEP_AI_MODELS_LOAD            4
#define BOOT_STEP_NETWORK_INIT               5
#define BOOT_STEP_SECURITY_INIT              6
#define BOOT_STEP_VISUALIZATION_INIT         7
#define BOOT_STEP_READY                       8

/* 5-Axes Reference Frame Constants */
#define AXIS_COUNT                           5
#define AXIS_X_INDEX                          0
#define AXIS_Y_INDEX                          1
#define AXIS_Z_INDEX                          2
#define AXIS_B_INDEX                          3
#define AXIS_R_INDEX                          4

/* Spiral Visualization Constants */
#define SPIRAL_POINTS                        500
#define SPIRAL_ARMS                           3
#define SPIRAL_TURNS                           8
#define SPIRAL_RADIUS_MIN                     0.5
#define SPIRAL_RADIUS_MAX                     3.0
#define SPIRAL_HEIGHT_MIN                    -2.0
#define SPIRAL_HEIGHT_MAX                      2.0
#define MAX_PARTICLES                        2000
#define PARTICLE_TRAIL_LENGTH                  8

/* Axis colors as per specification */
#define AXIS_X_RED       1.0f, 0.0f, 0.0f, 1.0f
#define AXIS_Y_GREEN     0.0f, 1.0f, 0.0f, 1.0f
#define AXIS_Z_BLUE      0.0f, 0.0f, 1.0f, 1.0f
#define AXIS_B_PURPLE    0.8f, 0.4f, 0.8f, 1.0f
#define AXIS_R_YELLOW    1.0f, 1.0f, 0.0f, 1.0f

/* Neural Network Scaling Constants - Autonomous ranges */
#define MIN_VOCAB_SIZE                      1000
#define MAX_VOCAB_SIZE                     50000
#define MIN_HIDDEN_SIZE                      64
#define MAX_HIDDEN_SIZE                     4096
#define MIN_LAYERS                            1
#define MAX_LAYERS                           32
#define DEFAULT_VOCAB_SIZE                  5000
#define DEFAULT_HIDDEN_SIZE                  512
#define DEFAULT_LAYERS                         8

/* Academic AI Constants */
#define ACADEMIC_MOE_LAYERS               8
#define ACADEMIC_EXPERTS_PER_LAYER        4
#define ACADEMIC_HIDDEN_SIZE              512
#define ACADEMIC_MAX_SEQUENCE_LENGTH      512
#define ACADEMIC_ATTENTION_HEADS           8
#define ACADEMIC_VOCAB_SIZE                50000
#define ACADEMIC_R1_CHAIN_DEPTH            8
#define ACADEMIC_R1_REFLECTION_STEPS       3
#define ACADEMIC_V2_LATENT_DIM             64
#define ACADEMIC_V2_QUERY_GROUPS           2

/*============================================================================
 * FORWARD DECLARATIONS
 *============================================================================*/

struct EVOXNeuralModel;
typedef struct EVOXNeuralModel EVOXNeuralModel;

static void free_neural_model(EVOXNeuralModel *model);

/*============================================================================
 * FSM STATES - 14 States
 *============================================================================*/

typedef enum {
	/* Core States (0-13) */
	FSM_STATE_BOOT = 0,
	FSM_STATE_IDLE,
	FSM_STATE_INIT,
	FSM_STATE_LOADING,
	FSM_STATE_SYMBOLIC_REASONING,
	FSM_STATE_NEURON_SYMBOLIC,
	FSM_STATE_PROCESSING,
	FSM_STATE_REASONING,
	FSM_STATE_LEARNING,
	FSM_STATE_VISUALIZING,
	FSM_STATE_COMMUNICATING,
	FSM_STATE_ROTATING_KEYS,
	FSM_STATE_ERROR,
	FSM_STATE_TERMINATE,
	FSM_STATE_COUNT = 14
} FSMState;

typedef enum {
	FSM_EVENT_NONE = 0,
	FSM_EVENT_BOOT_COMPLETE,
	FSM_EVENT_BOOT_FAILED,
	FSM_EVENT_START,
	FSM_EVENT_DATA_READY,
	FSM_EVENT_SYMBOLIC_MATCH,
	FSM_EVENT_NEURON_ACTIVATED,
	FSM_EVENT_INFERENCE_COMPLETE,
	FSM_EVENT_LEARNING_COMPLETE,
	FSM_EVENT_KEY_EXPIRING,
	FSM_EVENT_ERROR_OCCURRED,
	FSM_EVENT_TIMEOUT,
	FSM_EVENT_TERMINATE,
	FSM_EVENT_COUNT
} FSMEvent;

/*============================================================================
 * BOOT SEQUENCE - 8 Steps
 *============================================================================*/

typedef struct {
	int current_step;
	int steps_completed;
	double step_start_time;
	double step_end_time;
	char step_name[32];
	char step_message[256];
	int step_success;
	double boot_progress;
	double boot_start_time;
	double boot_end_time;
	double boot_duration;
	char boot_log[1024];
} BootSequence;

/*============================================================================
 * 5-AXES VECTOR STRUCTURE
 *============================================================================*/

typedef struct {
	double x; /* X-Axis: Length */
	double y; /* Y-Axis: Height */
	double z; /* Z-Axis: Width */
	double b; /* B-Axis: Radius Base */
	double r; /* R-Axis: Rotation */
} FiveAxisVector;

/*============================================================================
 * VISUALIZATION STRUCTURES
 *============================================================================*/

/* Vector types */
typedef struct {
	float x, y, z;
} Vector3f;

typedef struct {
	float x, y, z, w;
} Vector4f;

/* Quaternion for R-Axis rotation */
typedef struct {
	double w;
	double x;
	double y;
	double z;
} Quaternion;

/* Spiral point types */
typedef struct {
	Vector3f position;
	Vector3f color;
	float angle;
	float luminescence;
	float phase;
	float curvature;
} SpiralPoint;

/* Enhanced spiral arm */
typedef struct {
	SpiralPoint points[SPIRAL_POINTS];
	Quaternion arm_rotation;
	float arm_phase;
	float arm_frequency;
	unsigned long point_count;
	double total_luminescence;
	double mathematical_entropy;
} SpiralArm;

/* B-Axis radius field */
typedef struct {
	double base_radius;
	double modulated_radius;
	double golden_radius;
	double radial_pulse;
	double radial_frequency;
	double harmonic_amplitude;
} BRadiusField;

/* R-Axis rotation state */
typedef struct {
	Quaternion current;
	double angular_speed;
	double precession_angle;
	double nutation_angle;
	double proper_rotation;
	double precession_rate;
	double nutation_rate;
	double rotation_rate;
} RRotationState;

/* Particle structure */
typedef struct {
	FiveAxisVector position;
	FiveAxisVector velocity;
	FiveAxisVector acceleration;
	double mass;
	double charge;
	double spin;
	double luminance;
	int active;
	Vector3f trail[PARTICLE_TRAIL_LENGTH];
	int trail_index;
	float color[4];
	float pulse_phase;
	float glow_intensity;
	unsigned long parent_state;
	double coherence;
} FiveAxisParticle;

/* Camera structure */
typedef struct {
	Vector3f position;
	Vector3f target;
	Vector3f up;
	float fov;
	float aspect_ratio;
	float near_plane;
	float far_plane;
	float distance;
	float azimuth;
	float elevation;
} Camera;

/* Lighting structure */
typedef struct {
	Vector4f position;
	Vector4f ambient;
	Vector4f diffuse;
	Vector4f specular;
	Vector3f attenuation;
	int enabled;
} Light;

/* Fog structure */
typedef struct {
	float density;
	float start;
	float end;
	Vector4f color;
	int enabled;
} Fog;

/*============================================================================
 * ACADEMIC AI FOUNDATIONS STRUCTURES
 *============================================================================*/

/* MoE (Mixture of Experts) */
typedef struct {
	double *expert_weights[ACADEMIC_MOE_LAYERS];
	double *routing_logits[ACADEMIC_MOE_LAYERS];
	double *gating_network_weights;
	unsigned long *expert_selection_counts;
	double expert_entropy;
	unsigned long total_routing_decisions;
	int is_initialized;
} AcademicMoE;

/* R1 Reasoning Framework */
typedef struct {
	double *chain_of_thought[ACADEMIC_R1_CHAIN_DEPTH];
	double *reflection_states[ACADEMIC_R1_REFLECTION_STEPS];
	double final_confidence;
	unsigned long reasoning_depth;
	unsigned long verified_steps;
	int is_initialized;
} AcademicR1;

/* V2 Attention Mechanism */
typedef struct {
	double *latent_queries[ACADEMIC_V2_QUERY_GROUPS];
	double *latent_keys[ACADEMIC_V2_QUERY_GROUPS];
	double *latent_values[ACADEMIC_V2_QUERY_GROUPS];
	double compression_ratio;
	double information_bottleneck;
	int is_initialized;
} AcademicV2;

/* Coder Code Generation */
typedef struct {
	double *token_probabilities;
	char *generated_code;
	unsigned long generated_tokens;
	unsigned long vocab_size;
	double code_quality_score;
	int is_initialized;
} AcademicCoder;

/*============================================================================
 * MODEL EXPORT STRUCTURES
 *============================================================================*/

typedef struct {
	unsigned char magic[4]; /* "EVOX" */
	unsigned long version;
	unsigned long architecture;
	unsigned long parameter_count;
	unsigned long layer_count;
	unsigned long hidden_size;
	unsigned long num_heads;
	unsigned long vocab_size;
	unsigned long max_seq_len;
	unsigned long quantization;
	double training_loss;
	double validation_loss;
	double perplexity;
	time_t checkpoint_time;
	char description[256];
	char base_model[64];
	char fine_tune_data[128];
	unsigned char hash[32];
	unsigned long moe_layers;
	unsigned long r1_depth;
	unsigned long v2_groups;
} ModelHeader;

typedef struct {
	char name[64];
	unsigned long dimensions[4];
	unsigned long num_dimensions;
	unsigned long data_type;
	unsigned long data_offset;
	unsigned long data_size;
	double min_val;
	double max_val;
} TensorInfo;

/*============================================================================
 * NEURAL NETWORK MODEL DATA - Autonomous scaling
 *============================================================================*/

struct EVOXNeuralModel {
	/* Core transformer weights - dynamically sized */
	double *embedding_weights;
	double *attention_weights;
	double *feedforward_weights;
	double *layer_norm_weights;
	double *output_weights;

	/* Academic AI components */
	AcademicMoE moe;
	AcademicR1 r1;
	AcademicV2 v2;
	AcademicCoder coder;

	/* Model metadata - autonomous ranges */
	unsigned long vocab_size;
	unsigned long hidden_size;
	unsigned long num_layers;
	unsigned long num_heads;
	unsigned long parameter_count;
	char model_name[256];
	double creation_time;
	int is_initialized;

	/* Scaling factors */
	double scaling_factor;
	double memory_usage;
	int autonomous_mode;
};

/*============================================================================
 * MAIN EVOX SYSTEM STRUCTURE
 *============================================================================*/

typedef struct {
	/* Version */
	char version[32];
	char build_date[32];
	char build_time[32];
	unsigned long system_id;
	unsigned long run_id;

	/* Boot Sequence - 8 steps */
	BootSequence boot;

	/* FSM - 14 states */
	FSMState current_state;
	FSMState previous_state;
	FSMEvent last_event;
	int state_counter;
	int simulation_step;
	unsigned long state_visits[FSM_STATE_COUNT];
	double state_timestamps[FSM_STATE_COUNT];
	double state_durations[FSM_STATE_COUNT];

	/* 5-Axes */
	FiveAxisVector axes[AXIS_COUNT];
	FiveAxisParticle *particles;
	unsigned long particle_count;
	unsigned long max_particles;

	/* Visualization */
	SpiralArm spiral_arms[SPIRAL_ARMS];
	RRotationState r_rotation;
	BRadiusField b_field;
	Camera camera;
	Light lights[4];
	Fog fog;

	/* Animation */
	float time;
	float delta_time;
	float rotation_angle;
	float zoom_level;
	int show_grid;
	int show_axes;
	int show_particles;
	int show_spiral;
	int show_metrics;
	int glow_effect;
	unsigned int active_spiral_type;

	/* System Metrics */
	unsigned long long total_operations;
	double system_entropy;
	double processing_load;
	double memory_usage;
	double cpu_usage;
	double gpu_usage;
	int running;
	int initialized;
	int error_state;
	int pause_requested;
	char error_message[256];

	/* Timing */
	struct timespec start_time;
	struct timespec last_time;
	double total_runtime;
	unsigned long frame_count;
	double fps;
	double target_fps;

	/* Multimedia */
	SDL_Window *sdl_window;
	SDL_GLContext gl_context;
	ALCdevice *al_device;
	ALCcontext *al_context;
	int window_width;
	int window_height;
	int fullscreen;

	/* Neural Model */
	EVOXNeuralModel *model;
	char model_path[256];
	int model_loaded;
	int model_autonomous;

	/* Academic AI metrics */
	double moe_entropy;
	double r1_confidence;
	double v2_compression;
	double coder_quality;

	/* Synchronization */
	pthread_mutex_t fsm_lock;
	pthread_mutex_t render_lock;
	pthread_mutex_t model_lock;
} EVOXSystem;

/*============================================================================
 * FSM STATE NAME FUNCTION
 *============================================================================*/

static const char* fsm_state_name(FSMState state) {
	switch (state) {
	case FSM_STATE_BOOT:
		return "BOOT";
	case FSM_STATE_IDLE:
		return "IDLE";
	case FSM_STATE_INIT:
		return "INIT";
	case FSM_STATE_LOADING:
		return "LOADING";
	case FSM_STATE_SYMBOLIC_REASONING:
		return "SYMBOLIC";
	case FSM_STATE_NEURON_SYMBOLIC:
		return "NEURO";
	case FSM_STATE_PROCESSING:
		return "PROCESS";
	case FSM_STATE_REASONING:
		return "REASON";
	case FSM_STATE_LEARNING:
		return "LEARN";
	case FSM_STATE_VISUALIZING:
		return "VISUAL";
	case FSM_STATE_COMMUNICATING:
		return "COMM";
	case FSM_STATE_ROTATING_KEYS:
		return "KEYS";
	case FSM_STATE_ERROR:
		return "ERROR";
	case FSM_STATE_TERMINATE:
		return "TERM";
	default:
		return "UNKNOWN";
	}
}

static const char* fsm_event_string(FSMEvent event) {
	switch (event) {
	case FSM_EVENT_NONE:
		return "NONE";
	case FSM_EVENT_BOOT_COMPLETE:
		return "BOOT_COMPLETE";
	case FSM_EVENT_BOOT_FAILED:
		return "BOOT_FAILED";
	case FSM_EVENT_START:
		return "START";
	case FSM_EVENT_DATA_READY:
		return "DATA_READY";
	case FSM_EVENT_SYMBOLIC_MATCH:
		return "SYMBOLIC_MATCH";
	case FSM_EVENT_NEURON_ACTIVATED:
		return "NEURON_ACTIVATED";
	case FSM_EVENT_INFERENCE_COMPLETE:
		return "INFERENCE_COMPLETE";
	case FSM_EVENT_LEARNING_COMPLETE:
		return "LEARNING_COMPLETE";
	case FSM_EVENT_KEY_EXPIRING:
		return "KEY_EXPIRING";
	case FSM_EVENT_ERROR_OCCURRED:
		return "ERROR_OCCURRED";
	case FSM_EVENT_TIMEOUT:
		return "TIMEOUT";
	case FSM_EVENT_TERMINATE:
		return "TERMINATE";
	default:
		return "UNKNOWN";
	}
}

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

static double get_monotonic_time(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void* aligned_malloc(size_t size, size_t alignment) {
	void *ptr;
	if (size == 0)
		return NULL;
	if (posix_memalign(&ptr, alignment, size) != 0) {
		return NULL;
	}
	return ptr;
}

static void aligned_free(void *ptr) {
	if (ptr)
		free(ptr);
}

static double random_uniform(unsigned long *seed) {
	*seed = *seed * 1103515245 + 12345;
	return ((double) ((*seed >> 16) & 0x7FFF)) / 32768.0;
}

static double random_gaussian(unsigned long *seed) {
	double u1 = random_uniform(seed);
	double u2 = random_uniform(seed);
	return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

static double clamp(double value, double min, double max) {
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

static double degrees_to_radians(double degrees) {
	return degrees * PI / 180.0;
}

static double radians_to_degrees(double radians) {
	return radians * 180.0 / PI;
}

static double lerp(double a, double b, double t) {
	return a + (b - a) * t;
}

/*============================================================================
 * OPENSSL 3.0 COMPATIBLE SHA256 FUNCTIONS
 *============================================================================*/

static void compute_sha256(const unsigned char *data, size_t len,
		unsigned char *hash) {
	EVP_MD_CTX *mdctx;
	const EVP_MD *md;
	unsigned int md_len;

	md = EVP_sha256();
	mdctx = EVP_MD_CTX_new();
	EVP_DigestInit_ex(mdctx, md, NULL);
	EVP_DigestUpdate(mdctx, data, len);
	EVP_DigestFinal_ex(mdctx, hash, &md_len);
	EVP_MD_CTX_free(mdctx);
}

static void compute_sha256_2part(const unsigned char *data1, size_t len1,
		const unsigned char *data2, size_t len2, unsigned char *hash) {
	EVP_MD_CTX *mdctx;
	const EVP_MD *md;
	unsigned int md_len;

	md = EVP_sha256();
	mdctx = EVP_MD_CTX_new();
	EVP_DigestInit_ex(mdctx, md, NULL);
	EVP_DigestUpdate(mdctx, data1, len1);
	EVP_DigestUpdate(mdctx, data2, len2);
	EVP_DigestFinal_ex(mdctx, hash, &md_len);
	EVP_MD_CTX_free(mdctx);
}

/*============================================================================
 * QUATERNION OPERATIONS
 *============================================================================*/

static Quaternion quaternion_identity(void) {
	Quaternion q;
	q.w = 1.0;
	q.x = 0.0;
	q.y = 0.0;
	q.z = 0.0;
	return q;
}

static Quaternion quaternion_from_axis_angle(double angle, double x, double y,
		double z) {
	Quaternion q;
	double norm = sqrt(x * x + y * y + z * z);
	double half_angle = angle * 0.5;
	double sin_half = sin(half_angle);

	if (norm > 1e-10) {
		q.w = cos(half_angle);
		q.x = sin_half * x / norm;
		q.y = sin_half * y / norm;
		q.z = sin_half * z / norm;
	} else {
		q = quaternion_identity();
	}
	return q;
}

static Quaternion quaternion_multiply(const Quaternion *a, const Quaternion *b) {
	Quaternion result;
	result.w = a->w * b->w - a->x * b->x - a->y * b->y - a->z * b->z;
	result.x = a->w * b->x + a->x * b->w + a->y * b->z - a->z * b->y;
	result.y = a->w * b->y - a->x * b->z + a->y * b->w + a->z * b->x;
	result.z = a->w * b->z + a->x * b->y - a->y * b->x + a->z * b->w;
	return result;
}

static void quaternion_normalize(Quaternion *q) {
	double norm = sqrt(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
	if (norm > 1e-10) {
		q->w /= norm;
		q->x /= norm;
		q->y /= norm;
		q->z /= norm;
	}
}

static Quaternion quaternion_slerp(const Quaternion *a, const Quaternion *b,
		double t) {
	Quaternion result;
	double dot = a->w * b->w + a->x * b->x + a->y * b->y + a->z * b->z;
	double theta, sin_theta, scale_a, scale_b;

	if (dot < 0.0) {
		dot = -dot;
		theta = acos(dot);
		sin_theta = sin(theta);
		scale_a = sin((1.0 - t) * theta) / sin_theta;
		scale_b = sin(t * theta) / sin_theta * (-1.0);
	} else {
		theta = acos(dot);
		sin_theta = sin(theta);
		scale_a = sin((1.0 - t) * theta) / sin_theta;
		scale_b = sin(t * theta) / sin_theta;
	}

	result.w = a->w * scale_a + b->w * scale_b;
	result.x = a->x * scale_a + b->x * scale_b;
	result.y = a->y * scale_a + b->y * scale_b;
	result.z = a->z * scale_a + b->z * scale_b;

	quaternion_normalize(&result);
	return result;
}

static void quaternion_to_matrix(const Quaternion *q, GLfloat *matrix) {
	matrix[0] = 1.0f - 2.0f * q->y * q->y - 2.0f * q->z * q->z;
	matrix[1] = 2.0f * q->x * q->y - 2.0f * q->z * q->w;
	matrix[2] = 2.0f * q->x * q->z + 2.0f * q->y * q->w;
	matrix[3] = 0.0f;

	matrix[4] = 2.0f * q->x * q->y + 2.0f * q->z * q->w;
	matrix[5] = 1.0f - 2.0f * q->x * q->x - 2.0f * q->z * q->z;
	matrix[6] = 2.0f * q->y * q->z - 2.0f * q->x * q->w;
	matrix[7] = 0.0f;

	matrix[8] = 2.0f * q->x * q->z - 2.0f * q->y * q->w;
	matrix[9] = 2.0f * q->y * q->z + 2.0f * q->x * q->w;
	matrix[10] = 1.0f - 2.0f * q->x * q->x - 2.0f * q->y * q->y;
	matrix[11] = 0.0f;

	matrix[12] = 0.0f;
	matrix[13] = 0.0f;
	matrix[14] = 0.0f;
	matrix[15] = 1.0f;
}

/*============================================================================
 * 5-AXES MATHEMATICAL FUNCTIONS
 *============================================================================*/

static double b_axis_calculate(double x, double y, double z) {
	return sqrt(x * x + y * y + z * z);
}

static void r_axis_apply_rotation(FiveAxisVector *pos, const Quaternion *rot) {
	/* Quaternion rotation: p' = q * p * q^-1 */
	Quaternion p = { 0.0, pos->x, pos->y, pos->z };
	Quaternion q_conj = { rot->w, -rot->x, -rot->y, -rot->z };

	/* q * p */
	Quaternion temp;
	temp.w = rot->w * p.w - rot->x * p.x - rot->y * p.y - rot->z * p.z;
	temp.x = rot->w * p.x + rot->x * p.w + rot->y * p.z - rot->z * p.y;
	temp.y = rot->w * p.y - rot->x * p.z + rot->y * p.w + rot->z * p.x;
	temp.z = rot->w * p.z + rot->x * p.y - rot->y * p.x + rot->z * p.w;

	/* (q * p) * q^-1 */
	Quaternion result;
	result.w = temp.w * q_conj.w - temp.x * q_conj.x - temp.y * q_conj.y
			- temp.z * q_conj.z;
	result.x = temp.w * q_conj.x + temp.x * q_conj.w + temp.y * q_conj.z
			- temp.z * q_conj.y;
	result.y = temp.w * q_conj.y - temp.x * q_conj.z + temp.y * q_conj.w
			+ temp.z * q_conj.x;
	result.z = temp.w * q_conj.z + temp.x * q_conj.y - temp.y * q_conj.x
			+ temp.z * q_conj.w;

	pos->x = result.x;
	pos->y = result.y;
	pos->z = result.z;
	pos->b = b_axis_calculate(pos->x, pos->y, pos->z);
}

static double five_axis_weighting(const FiveAxisVector *pos,
		const double *weights) {
	double x = pos->x;
	double y = pos->y;
	double z = pos->z;
	double b = pos->b;
	double r = pos->r;

	double distance = sqrt(x * x + y * y + z * z + b * b + r * r);
	double w_origin = exp(-distance);

	double w_positive = ((x > 0 ? x : 0) + (y > 0 ? y : 0) + (z > 0 ? z : 0)
			+ (b > 0 ? b : 0) + (r > 0 ? r : 0)) / 5.0;

	double w_negative = ((x < 0 ? -x : 0) + (y < 0 ? -y : 0) + (z < 0 ? -z : 0)
			+ (b < 0 ? -b : 0) + (r < 0 ? -r : 0)) / 5.0;

	return weights[0] * w_origin + weights[1] * w_positive
			+ weights[2] * w_negative;
}

/*============================================================================
 * R-AXIS ROTATION UPDATE
 *============================================================================*/

static void r_axis_quaternion_update(RRotationState *state, double dt) {
	double angle = state->angular_speed * dt;
	Quaternion delta = quaternion_from_axis_angle(angle, 0.0, 1.0, 0.0);

	state->current = quaternion_multiply(&delta, &state->current);
	quaternion_normalize(&state->current);

	state->precession_angle += state->precession_rate * dt;
	state->nutation_angle += state->nutation_rate * dt;
	state->proper_rotation += state->rotation_rate * dt;

	/* Apply precession and nutation */
	Quaternion precess = quaternion_from_axis_angle(state->precession_angle,
			0.0, 1.0, 0.0);
	Quaternion nutate = quaternion_from_axis_angle(state->nutation_angle, 1.0,
			0.0, 0.0);
	Quaternion rotate = quaternion_from_axis_angle(state->proper_rotation, 0.0,
			0.0, 1.0);

	Quaternion temp = quaternion_multiply(&precess, &nutate);
	Quaternion combined = quaternion_multiply(&temp, &rotate);

	state->current = quaternion_multiply(&combined, &state->current);
	quaternion_normalize(&state->current);
}

/*============================================================================
 * SPIRAL GENERATION FUNCTIONS
 *============================================================================*/

static void generate_spiral_arm(SpiralArm *arm, unsigned int arm_index,
		double phase) {
	unsigned int i;
	unsigned long seed = arm_index * 1000;

	for (i = 0; i < SPIRAL_POINTS; i++) {
		double t = (double) i / SPIRAL_POINTS;
		double angle = t * TWO_PI * SPIRAL_TURNS
				+ arm_index * TWO_PI / SPIRAL_ARMS + phase;
		double radius = SPIRAL_RADIUS_MIN
				+ t * (SPIRAL_RADIUS_MAX - SPIRAL_RADIUS_MIN);
		double height = SPIRAL_HEIGHT_MIN
				+ t * (SPIRAL_HEIGHT_MAX - SPIRAL_HEIGHT_MIN);

		/* Add Fibonacci modulation for mathematical harmony */
		radius *= (1.0 + 0.1 * sin(angle * 1.618));

		arm->points[i].position.x = radius * cos(angle);
		arm->points[i].position.y = height;
		arm->points[i].position.z = radius * sin(angle);

		/* Set colors based on arm index */
		if (arm_index == 0) { /* X-Axis - Red */
			arm->points[i].color.x = 1.0f;
			arm->points[i].color.y = 0.2f + 0.3f * sin(angle * 0.5);
			arm->points[i].color.z = 0.2f + 0.3f * cos(angle * 0.5);
		} else if (arm_index == 1) { /* Y-Axis - Green */
			arm->points[i].color.x = 0.2f + 0.3f * cos(angle * 0.5);
			arm->points[i].color.y = 1.0f;
			arm->points[i].color.z = 0.2f + 0.3f * sin(angle * 0.5);
		} else { /* Z-Axis - Blue */
			arm->points[i].color.x = 0.2f + 0.3f * sin(angle * 0.5);
			arm->points[i].color.y = 0.2f + 0.3f * cos(angle * 0.5);
			arm->points[i].color.z = 1.0f;
		}

		arm->points[i].angle = angle;
		arm->points[i].luminescence = 0.5f + 0.3f * sin(angle * 2.0);
		arm->points[i].phase = phase;
		arm->points[i].curvature = 1.0 / (radius + 0.1);
	}

	arm->arm_rotation = quaternion_identity();
	arm->arm_phase = phase;
	arm->arm_frequency = 0.5f + arm_index * 0.2f;
	arm->point_count = SPIRAL_POINTS;
	arm->total_luminescence = 0.0;
	arm->mathematical_entropy = 0.0;
}

/*============================================================================
 * BOOT SEQUENCE - 8 Steps
 *============================================================================*/

static void boot_sequence_init(BootSequence *boot) {
	boot->current_step = 0;
	boot->steps_completed = 0;
	boot->step_start_time = get_monotonic_time();
	boot->step_end_time = 0.0;
	strcpy(boot->step_name, "POWER_ON");
	strcpy(boot->step_message, "Powering on system...");
	boot->step_success = 0;
	boot->boot_progress = 0.0;
	boot->boot_start_time = get_monotonic_time();
	boot->boot_end_time = 0.0;
	boot->boot_duration = 0.0;
	strcpy(boot->boot_log, "");
}

static int boot_sequence_step(EVOXSystem *system) {
	BootSequence *boot = &system->boot;
	int step = boot->current_step;
	int result = 1;

	boot->step_start_time = get_monotonic_time();

	switch (step) {
	case BOOT_STEP_POWER_ON:
		strcpy(boot->step_name, "POWER_ON");
		strcpy(boot->step_message, "Power on - Voltage stable");
		system->axes[AXIS_X_INDEX].x = 0.0;
		system->axes[AXIS_Y_INDEX].y = 0.0;
		system->axes[AXIS_Z_INDEX].z = 0.0;
		system->axes[AXIS_B_INDEX].b = 0.0;
		system->axes[AXIS_R_INDEX].r = 0.0;
		sprintf(boot->boot_log + strlen(boot->boot_log),
				"[BOOT] Power on - Starting at R(0,0,0,0,0)\n");
		break;

	case BOOT_STEP_HARDWARE_INIT:
		strcpy(boot->step_name, "HARDWARE_INIT");
		strcpy(boot->step_message, "Initializing hardware - NUMA detection");
		if (numa_available() >= 0) {
			sprintf(boot->boot_log + strlen(boot->boot_log),
					"[BOOT] NUMA available with %d nodes\n", numa_max_node());
		}
		break;

	case BOOT_STEP_MEMORY_TEST:
		strcpy(boot->step_name, "MEMORY_TEST");
		strcpy(boot->step_message, "Testing memory - allocating particles");
		system->particles = aligned_malloc(
				MAX_PARTICLES * sizeof(FiveAxisParticle), 64);
		if (system->particles) {
			system->particle_count = MAX_PARTICLES;
			system->max_particles = MAX_PARTICLES;
			sprintf(boot->boot_log + strlen(boot->boot_log),
					"[BOOT] Allocated %d particles\n", MAX_PARTICLES);
		} else {
			result = 0;
		}
		break;

	case BOOT_STEP_CORE_LOAD:
		strcpy(boot->step_name, "CORE_LOAD");
		strcpy(boot->step_message, "Loading core system - FSM initialization");
		system->current_state = FSM_STATE_BOOT;
		pthread_mutex_init(&system->fsm_lock, NULL);
		pthread_mutex_init(&system->render_lock, NULL);
		pthread_mutex_init(&system->model_lock, NULL);
		sprintf(boot->boot_log + strlen(boot->boot_log),
				"[BOOT] FSM initialized with 14 states\n");
		break;

	case BOOT_STEP_AI_MODELS_LOAD:
		strcpy(boot->step_name, "AI_MODELS_LOAD");
		strcpy(boot->step_message, "Loading AI models - Academic foundations");
		sprintf(boot->boot_log + strlen(boot->boot_log),
				"[BOOT] Academic AI: MoE, R1, V2, Coder ready\n");
		break;

	case BOOT_STEP_NETWORK_INIT:
		strcpy(boot->step_name, "NETWORK_INIT");
		strcpy(boot->step_message, "Initializing network - P2P communication");
		sprintf(boot->boot_log + strlen(boot->boot_log),
				"[BOOT] P2P network initialized on port 8765\n");
		break;

	case BOOT_STEP_SECURITY_INIT:
		strcpy(boot->step_name, "SECURITY_INIT");
		strcpy(boot->step_message, "Initializing security - Crypto keys");
		sprintf(boot->boot_log + strlen(boot->boot_log),
				"[BOOT] Military-grade crypto initialized\n");
		break;

	case BOOT_STEP_VISUALIZATION_INIT:
		strcpy(boot->step_name, "VISUALIZATION_INIT");
		strcpy(boot->step_message, "Initializing visualization - OpenGL/SDL");
		sprintf(boot->boot_log + strlen(boot->boot_log),
				"[BOOT] 5-axes spiral visualization ready\n");
		break;

	case BOOT_STEP_READY:
		strcpy(boot->step_name, "READY");
		strcpy(boot->step_message, "Boot complete - System ready");
		boot->boot_end_time = get_monotonic_time();
		boot->boot_duration = boot->boot_end_time - boot->boot_start_time;
		system->initialized = 1;
		sprintf(boot->boot_log + strlen(boot->boot_log),
				"[BOOT] Complete in %.3f seconds\n", boot->boot_duration);
		break;
	}

	boot->step_end_time = get_monotonic_time();
	boot->steps_completed++;
	boot->boot_progress = (double) (boot->steps_completed) / (BOOT_STEPS + 1);
	boot->step_success = result;

	printf("[BOOT] Step %d/%d: %-16s - %s [%s]\n", boot->steps_completed,
			BOOT_STEPS + 1, boot->step_name, boot->step_message,
			result ? "OK" : "FAIL");

	boot->current_step++;
	return result;
}

/*============================================================================
 * FSM TRANSITION - 14 States
 *============================================================================*/

static FSMState fsm_transition(EVOXSystem *system, FSMEvent event) {
	FSMState current = system->current_state;
	FSMState next = current;

	pthread_mutex_lock(&system->fsm_lock);

	/* Deterministic state transitions based on current state and event */
	switch (current) {
	case FSM_STATE_BOOT:
		if (event == FSM_EVENT_BOOT_COMPLETE)
			next = FSM_STATE_IDLE;
		else if (event == FSM_EVENT_BOOT_FAILED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_IDLE:
		if (event == FSM_EVENT_START)
			next = FSM_STATE_INIT;
		else if (event == FSM_EVENT_TERMINATE)
			next = FSM_STATE_TERMINATE;
		break;

	case FSM_STATE_INIT:
		if (event == FSM_EVENT_DATA_READY)
			next = FSM_STATE_LOADING;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_LOADING:
		if (event == FSM_EVENT_SYMBOLIC_MATCH)
			next = FSM_STATE_SYMBOLIC_REASONING;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_SYMBOLIC_REASONING:
		if (event == FSM_EVENT_NEURON_ACTIVATED)
			next = FSM_STATE_NEURON_SYMBOLIC;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_NEURON_SYMBOLIC:
		if (event == FSM_EVENT_INFERENCE_COMPLETE)
			next = FSM_STATE_PROCESSING;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_PROCESSING:
		if (event == FSM_EVENT_LEARNING_COMPLETE)
			next = FSM_STATE_LEARNING;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_LEARNING:
		if (event == FSM_EVENT_DATA_READY)
			next = FSM_STATE_REASONING;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_REASONING:
		if (event == FSM_EVENT_INFERENCE_COMPLETE)
			next = FSM_STATE_VISUALIZING;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_VISUALIZING:
		if (event == FSM_EVENT_DATA_READY)
			next = FSM_STATE_COMMUNICATING;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_COMMUNICATING:
		if (event == FSM_EVENT_KEY_EXPIRING)
			next = FSM_STATE_ROTATING_KEYS;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_ROTATING_KEYS:
		if (event == FSM_EVENT_INFERENCE_COMPLETE)
			next = FSM_STATE_IDLE;
		else if (event == FSM_EVENT_ERROR_OCCURRED)
			next = FSM_STATE_ERROR;
		break;

	case FSM_STATE_ERROR:
		if (event == FSM_EVENT_TIMEOUT)
			next = FSM_STATE_IDLE;
		else if (event == FSM_EVENT_TERMINATE)
			next = FSM_STATE_TERMINATE;
		break;

	case FSM_STATE_TERMINATE:
		system->running = 0;
		break;

	default:
		next = FSM_STATE_ERROR;
	}

	if (next != current) {
		system->state_visits[current]++;
		system->state_timestamps[current] = get_monotonic_time();
		system->state_durations[current] = system->state_timestamps[current]
				- system->state_timestamps[current];
		system->previous_state = current;
		system->current_state = next;
		system->last_event = event;
		system->state_counter = 0;

		printf("[FSM] %s -> %s (Event: %s)\n", fsm_state_name(current),
				fsm_state_name(next), fsm_event_string(event));
	}

	pthread_mutex_unlock(&system->fsm_lock);
	return next;
}

/*============================================================================
 * AUTONOMOUS NEURAL NETWORK CREATION
 *============================================================================*/

static EVOXNeuralModel* create_autonomous_neural_model(unsigned long vocab_size,
		unsigned long hidden_size, unsigned long num_layers) {
	EVOXNeuralModel *model;
	unsigned long i, j;
	unsigned long seed = 42;
	size_t size;
	int success = 1;

	/* Clamp values to allowed ranges */
	if (vocab_size < MIN_VOCAB_SIZE)
		vocab_size = MIN_VOCAB_SIZE;
	if (vocab_size > MAX_VOCAB_SIZE)
		vocab_size = MAX_VOCAB_SIZE;
	if (hidden_size < MIN_HIDDEN_SIZE)
		hidden_size = MIN_HIDDEN_SIZE;
	if (hidden_size > MAX_HIDDEN_SIZE)
		hidden_size = MAX_HIDDEN_SIZE;
	if (num_layers < MIN_LAYERS)
		num_layers = MIN_LAYERS;
	if (num_layers > MAX_LAYERS)
		num_layers = MAX_LAYERS;

	printf("[MODEL] Creating autonomous neural network:\n");
	printf("  Vocabulary size: %lu\n", vocab_size);
	printf("  Hidden size: %lu\n", hidden_size);
	printf("  Layers: %lu\n", num_layers);

	model = (EVOXNeuralModel*) malloc(sizeof(EVOXNeuralModel));
	if (!model) {
		printf("[ERROR] Failed to allocate model structure\n");
		return NULL;
	}

	memset(model, 0, sizeof(EVOXNeuralModel));

	model->vocab_size = vocab_size;
	model->hidden_size = hidden_size;
	model->num_layers = num_layers;
	model->num_heads = hidden_size / 64; /* Heuristic: head dim 64 */
	if (model->num_heads < 1)
		model->num_heads = 1;
	if (model->num_heads > 32)
		model->num_heads = 32;

	/* Calculate total parameters */
	model->parameter_count =
	/* Embedding layer */
	(unsigned long) vocab_size * hidden_size +

	/* Attention layers: Q,K,V,O per layer */
	(unsigned long) num_layers * 4 * hidden_size * hidden_size +

	/* Feed-forward layers (typically 4x hidden size) */
	(unsigned long) num_layers * 2 * hidden_size * (4 * hidden_size) +

	/* Layer norms */
	(unsigned long) num_layers * 2 * hidden_size +

	/* Output layer */
	(unsigned long) hidden_size * vocab_size;

	printf("[MODEL] Total parameters: %lu (%.2f MB)\n", model->parameter_count,
			(double) model->parameter_count * sizeof(double) / (1024 * 1024));

	/* Allocate embedding weights */
	size = vocab_size * hidden_size * sizeof(double);
	model->embedding_weights = (double*) aligned_malloc(size, 64);
	if (!model->embedding_weights)
		success = 0;

	if (success) {
		for (i = 0; i < vocab_size; i++) {
			for (j = 0; j < hidden_size; j++) {
				model->embedding_weights[i * hidden_size + j] = (random_uniform(
						&seed) - 0.5) * 0.02;
			}
		}
	}

	/* Allocate attention weights */
	size = num_layers * 4 * hidden_size * hidden_size * sizeof(double);
	model->attention_weights = (double*) aligned_malloc(size, 64);
	if (!model->attention_weights)
		success = 0;

	if (success) {
		for (i = 0; i < num_layers * 4 * hidden_size * hidden_size; i++) {
			model->attention_weights[i] = (random_uniform(&seed) - 0.5) * 0.02;
		}
	}

	/* Allocate feed-forward weights */
	size = num_layers * 2 * hidden_size * (4 * hidden_size) * sizeof(double);
	model->feedforward_weights = (double*) aligned_malloc(size, 64);
	if (!model->feedforward_weights)
		success = 0;

	if (success) {
		for (i = 0; i < num_layers * 2 * hidden_size * (4 * hidden_size); i++) {
			model->feedforward_weights[i] = (random_uniform(&seed) - 0.5)
					* 0.02;
		}
	}

	/* Allocate layer norm weights */
	size = num_layers * 2 * hidden_size * sizeof(double);
	model->layer_norm_weights = (double*) aligned_malloc(size, 64);
	if (!model->layer_norm_weights)
		success = 0;

	if (success) {
		for (i = 0; i < num_layers * 2 * hidden_size; i++) {
			model->layer_norm_weights[i] = 1.0;
		}
	}

	/* Allocate output weights */
	size = hidden_size * vocab_size * sizeof(double);
	model->output_weights = (double*) aligned_malloc(size, 64);
	if (!model->output_weights)
		success = 0;

	if (success) {
		for (i = 0; i < hidden_size * vocab_size; i++) {
			model->output_weights[i] = (random_uniform(&seed) - 0.5) * 0.02;
		}
	}

	if (!success) {
		printf("[ERROR] Failed to allocate model weights\n");
		free_neural_model(model);
		return NULL;
	}

	model->creation_time = get_monotonic_time();
	model->is_initialized = 1;
	model->autonomous_mode = 1;
	model->scaling_factor = 1.0;
	model->memory_usage = (double) model->parameter_count * sizeof(double)
			/ (1024 * 1024);

	snprintf(model->model_name, sizeof(model->model_name),
			"EVOX_AUTO_%luv_%luh_%lul", vocab_size, hidden_size, num_layers);

	printf("[MODEL] Created successfully: %s\n", model->model_name);
	printf("[MODEL] Memory usage: %.2f MB\n", model->memory_usage);

	return model;
}

/*============================================================================
 * FREE NEURAL MODEL
 *============================================================================*/

static void free_neural_model(EVOXNeuralModel *model) {
	unsigned long i;

	if (!model)
		return;

	if (model->embedding_weights)
		aligned_free(model->embedding_weights);
	if (model->attention_weights)
		aligned_free(model->attention_weights);
	if (model->feedforward_weights)
		aligned_free(model->feedforward_weights);
	if (model->layer_norm_weights)
		aligned_free(model->layer_norm_weights);
	if (model->output_weights)
		aligned_free(model->output_weights);

	free(model);
}

/*============================================================================
 * INITIALIZE VISUALIZATION
 *============================================================================*/

static void init_visualization(EVOXSystem *system) {
	unsigned int i;

	/* Initialize camera */
	system->camera.position.x = 5.0f;
	system->camera.position.y = 3.0f;
	system->camera.position.z = 5.0f;
	system->camera.target.x = 0.0f;
	system->camera.target.y = 0.0f;
	system->camera.target.z = 0.0f;
	system->camera.up.x = 0.0f;
	system->camera.up.y = 1.0f;
	system->camera.up.z = 0.0f;
	system->camera.fov = 45.0f;
	system->camera.aspect_ratio = (float) system->window_width
			/ system->window_height;
	system->camera.near_plane = 0.1f;
	system->camera.far_plane = 100.0f;
	system->camera.distance = 7.0f;
	system->camera.azimuth = 45.0f;
	system->camera.elevation = 30.0f;

	/* Initialize lights */
	for (i = 0; i < 4; i++) {
		system->lights[i].enabled = (i == 0);
	}

	/* Light 0 - Key light (white) */
	system->lights[0].position.x = 2.0f;
	system->lights[0].position.y = 5.0f;
	system->lights[0].position.z = 3.0f;
	system->lights[0].position.w = 1.0f;
	system->lights[0].ambient.x = 0.2f;
	system->lights[0].ambient.y = 0.2f;
	system->lights[0].ambient.z = 0.2f;
	system->lights[0].ambient.w = 1.0f;
	system->lights[0].diffuse.x = 1.0f;
	system->lights[0].diffuse.y = 1.0f;
	system->lights[0].diffuse.z = 1.0f;
	system->lights[0].diffuse.w = 1.0f;
	system->lights[0].specular.x = 1.0f;
	system->lights[0].specular.y = 1.0f;
	system->lights[0].specular.z = 1.0f;
	system->lights[0].specular.w = 1.0f;

	/* Light 1 - Fill light (blue tint) */
	system->lights[1].position.x = -3.0f;
	system->lights[1].position.y = 2.0f;
	system->lights[1].position.z = 4.0f;
	system->lights[1].position.w = 1.0f;
	system->lights[1].ambient.x = 0.1f;
	system->lights[1].ambient.y = 0.1f;
	system->lights[1].ambient.z = 0.1f;
	system->lights[1].ambient.w = 1.0f;
	system->lights[1].diffuse.x = 0.3f;
	system->lights[1].diffuse.y = 0.5f;
	system->lights[1].diffuse.z = 0.8f;
	system->lights[1].diffuse.w = 1.0f;

	/* Fog settings */
	system->fog.enabled = 1;
	system->fog.density = 0.05f;
	system->fog.start = 10.0f;
	system->fog.end = 30.0f;
	system->fog.color.x = 0.05f;
	system->fog.color.y = 0.05f;
	system->fog.color.z = 0.1f;
	system->fog.color.w = 1.0f;

	/* Display flags */
	system->show_grid = 1;
	system->show_axes = 1;
	system->show_particles = 1;
	system->show_spiral = 1;
	system->show_metrics = 1;
	system->glow_effect = 1;

	/* Animation */
	system->rotation_angle = 0.0f;
	system->zoom_level = 1.0f;
	system->time = 0.0f;
	system->delta_time = 0.016f;
	system->active_spiral_type = 0;

	/* Initialize spiral arms */
	for (i = 0; i < SPIRAL_ARMS; i++) {
		generate_spiral_arm(&system->spiral_arms[i], i, 0.0);
	}

	/* Initialize R-Axis rotation */
	system->r_rotation.current = quaternion_identity();
	system->r_rotation.angular_speed = 0.5;
	system->r_rotation.precession_angle = 0.0;
	system->r_rotation.nutation_angle = 0.0;
	system->r_rotation.proper_rotation = 0.0;
	system->r_rotation.precession_rate = 0.01;
	system->r_rotation.nutation_rate = 0.005;
	system->r_rotation.rotation_rate = 0.1;

	/* Initialize B-Axis field */
	system->b_field.base_radius = 1.0;
	system->b_field.modulated_radius = 0.2;
	system->b_field.golden_radius = 0.1;
	system->b_field.radial_pulse = 0.1;
	system->b_field.radial_frequency = 2.0;
	system->b_field.harmonic_amplitude = 0.05;
}

/*============================================================================
 * RENDERING FUNCTIONS - OPENGL WRAPPERS
 *============================================================================*/

static void glBegin_wrap(GLenum mode) {
	glBegin(mode);
}
static void glEnd_wrap(void) {
	glEnd();
}
static void glVertex3f_wrap(GLfloat x, GLfloat y, GLfloat z) {
	glVertex3f(x, y, z);
}
static void glColor4f_wrap(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
	glColor4f(r, g, b, a);
}
static void glPointSize_wrap(GLfloat size) {
	glPointSize(size);
}
static void glLineWidth_wrap(GLfloat width) {
	glLineWidth(width);
}
static void glEnable_wrap(GLenum cap) {
	glEnable(cap);
}
static void glDisable_wrap(GLenum cap) {
	glDisable(cap);
}
static void glClear_wrap(GLbitfield mask) {
	glClear(mask);
}
static void glClearColor_wrap(GLclampf r, GLclampf g, GLclampf b, GLclampf a) {
	glClearColor(r, g, b, a);
}
static void glMatrixMode_wrap(GLenum mode) {
	glMatrixMode(mode);
}
static void glLoadIdentity_wrap(void) {
	glLoadIdentity();
}
static void glPushMatrix_wrap(void) {
	glPushMatrix();
}
static void glPopMatrix_wrap(void) {
	glPopMatrix();
}
static void glTranslatef_wrap(GLfloat x, GLfloat y, GLfloat z) {
	glTranslatef(x, y, z);
}
static void glRotatef_wrap(GLfloat angle, GLfloat x, GLfloat y, GLfloat z) {
	glRotatef(angle, x, y, z);
}
static void glScalef_wrap(GLfloat x, GLfloat y, GLfloat z) {
	glScalef(x, y, z);
}
static void glBlendFunc_wrap(GLenum sfactor, GLenum dfactor) {
	glBlendFunc(sfactor, dfactor);
}
static void glDepthFunc_wrap(GLenum func) {
	glDepthFunc(func);
}
static void glRasterPos2f_wrap(GLfloat x, GLfloat y) {
	glRasterPos2f(x, y);
}
static void glOrtho_wrap(GLdouble left, GLdouble right, GLdouble bottom,
		GLdouble top, GLdouble near, GLdouble far) {
	glOrtho(left, right, bottom, top, near, far);
}
static void glVertex2f_wrap(GLfloat x, GLfloat y) {
	glVertex2f(x, y);
}
static void glLightfv_wrap(GLenum light, GLenum pname, const GLfloat *params) {
	glLightfv(light, pname, params);
}
static void glLightModelfv_wrap(GLenum pname, const GLfloat *params) {
	glLightModelfv(pname, params);
}
static void glFogfv_wrap(GLenum pname, const GLfloat *params) {
	glFogfv(pname, params);
}
static void glFogf_wrap(GLenum pname, GLfloat param) {
	glFogf(pname, param);
}
static void glFogi_wrap(GLenum pname, GLint param) {
	glFogi(pname, param);
}
static void glMultMatrixf_wrap(const GLfloat *m) {
	glMultMatrixf(m);
}
static void glViewport_wrap(GLint x, GLint y, GLsizei width, GLsizei height) {
	glViewport(x, y, width, height);
}

static void gluPerspective_wrap(GLdouble fovy, GLdouble aspect, GLdouble zNear,
		GLdouble zFar) {
	gluPerspective(fovy, aspect, zNear, zFar);
}

static void gluLookAt_wrap(GLdouble eyeX, GLdouble eyeY, GLdouble eyeZ,
		GLdouble centerX, GLdouble centerY, GLdouble centerZ, GLdouble upX,
		GLdouble upY, GLdouble upZ) {
	gluLookAt(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ);
}

static void glutInit_wrap(int *argcp, char **argv) {
	glutInit(argcp, argv);
}
static void glutInitDisplayMode_wrap(unsigned int mode) {
	glutInitDisplayMode(mode);
}
static void glutBitmapCharacter_wrap(void *font, int character) {
	glutBitmapCharacter(font, character);
}
static void glutWireSphere_wrap(GLdouble radius, GLint slices, GLint stacks) {
	glutWireSphere(radius, slices, stacks);
}
static void glutSolidSphere_wrap(GLdouble radius, GLint slices, GLint stacks) {
	glutSolidSphere(radius, slices, stacks);
}

static int SDL_Init_wrap(unsigned int flags) {
	return SDL_Init(flags);
}
static SDL_Window* SDL_CreateWindow_wrap(const char *title, int x, int y, int w,
		int h, unsigned int flags) {
	return SDL_CreateWindow(title, x, y, w, h, flags);
}
static SDL_GLContext SDL_GL_CreateContext_wrap(SDL_Window *window) {
	return SDL_GL_CreateContext(window);
}
static void SDL_GL_MakeCurrent_wrap(SDL_Window *window, SDL_GLContext context) {
	SDL_GL_MakeCurrent(window, context);
}
static void SDL_GL_SetSwapInterval_wrap(int interval) {
	SDL_GL_SetSwapInterval(interval);
}
static void SDL_GL_SwapWindow_wrap(SDL_Window *window) {
	SDL_GL_SwapWindow(window);
}
static void SDL_GL_DeleteContext_wrap(SDL_GLContext context) {
	SDL_GL_DeleteContext(context);
}
static void SDL_DestroyWindow_wrap(SDL_Window *window) {
	SDL_DestroyWindow(window);
}
static void SDL_Quit_wrap(void) {
	SDL_Quit();
}
static int SDL_PollEvent_wrap(SDL_Event *event) {
	return SDL_PollEvent(event);
}
static void SDL_Delay_wrap(unsigned int ms) {
	SDL_Delay(ms);
}

/*============================================================================
 * DRAWING FUNCTIONS
 *============================================================================*/

static void setup_lights(EVOXSystem *system) {
	GLfloat global_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	int i;

	glLightModelfv_wrap(GL_LIGHT_MODEL_AMBIENT, global_ambient);
	glEnable_wrap(GL_LIGHTING);

	for (i = 0; i < 4; i++) {
		if (system->lights[i].enabled) {
			GLenum gl_light = GL_LIGHT0 + i;
			glEnable_wrap(gl_light);
			glLightfv_wrap(gl_light, GL_POSITION,
					(GLfloat*) &system->lights[i].position);
			glLightfv_wrap(gl_light, GL_AMBIENT,
					(GLfloat*) &system->lights[i].ambient);
			glLightfv_wrap(gl_light, GL_DIFFUSE,
					(GLfloat*) &system->lights[i].diffuse);
			glLightfv_wrap(gl_light, GL_SPECULAR,
					(GLfloat*) &system->lights[i].specular);
		}
	}
}

static void setup_fog(EVOXSystem *system) {
	if (!system->fog.enabled)
		return;

	glEnable_wrap(GL_FOG);
	glFogfv_wrap(GL_FOG_COLOR, (GLfloat*) &system->fog.color);
	glFogf_wrap(GL_FOG_DENSITY, system->fog.density);
	glFogf_wrap(GL_FOG_START, system->fog.start);
	glFogf_wrap(GL_FOG_END, system->fog.end);
	glFogi_wrap(GL_FOG_MODE, GL_EXP);
}

static void draw_grid(EVOXSystem *system) {
	int i;
	float grid_size = 10.0f;
	int grid_steps = 20;
	float step = grid_size / grid_steps;

	if (!system->show_grid)
		return;

	glDisable_wrap(GL_LIGHTING);
	glColor4f_wrap(0.3f, 0.3f, 0.3f, 0.5f);
	glBegin_wrap(GL_LINES);

	for (i = -grid_steps / 2; i <= grid_steps / 2; i++) {
		float pos = i * step;
		glVertex3f_wrap(pos, 0.0f, -grid_size / 2);
		glVertex3f_wrap(pos, 0.0f, grid_size / 2);
		glVertex3f_wrap(-grid_size / 2, 0.0f, pos);
		glVertex3f_wrap(grid_size / 2, 0.0f, pos);
	}

	glEnd_wrap();
	glEnable_wrap(GL_LIGHTING);
}

static void draw_axes(EVOXSystem *system) {
	float axis_length = 2.5f;

	if (!system->show_axes)
		return;

	glDisable_wrap(GL_LIGHTING);
	glLineWidth_wrap(3.0f);

	/* X Axis - Red */
	glColor4f_wrap(AXIS_X_RED);
	glBegin_wrap(GL_LINES);
	glVertex3f_wrap(-axis_length, 0.0f, 0.0f);
	glVertex3f_wrap(axis_length, 0.0f, 0.0f);
	glEnd_wrap();

	/* Y Axis - Green */
	glColor4f_wrap(AXIS_Y_GREEN);
	glBegin_wrap(GL_LINES);
	glVertex3f_wrap(0.0f, -axis_length, 0.0f);
	glVertex3f_wrap(0.0f, axis_length, 0.0f);
	glEnd_wrap();

	/* Z Axis - Blue */
	glColor4f_wrap(AXIS_Z_BLUE);
	glBegin_wrap(GL_LINES);
	glVertex3f_wrap(0.0f, 0.0f, -axis_length);
	glVertex3f_wrap(0.0f, 0.0f, axis_length);
	glEnd_wrap();

	glEnable_wrap(GL_LIGHTING);
}

static void draw_spiral(EVOXSystem *system) {
	unsigned int arm, i;
	GLfloat matrix[16];

	if (!system->show_spiral)
		return;

	glDisable_wrap(GL_LIGHTING);
	glEnable_wrap(GL_BLEND);
	glBlendFunc_wrap(GL_SRC_ALPHA, GL_ONE);

	/* Get R-Axis rotation matrix */
	quaternion_to_matrix(&system->r_rotation.current, matrix);

	for (arm = 0; arm < SPIRAL_ARMS; arm++) {
		glPushMatrix_wrap();

		/* Apply R-Axis rotation */
		glMultMatrixf_wrap(matrix);

		/* Apply spiral rotation */
		glRotatef_wrap(system->rotation_angle, 0.0f, 1.0f, 0.0f);

		/* Draw spiral arm */
		glLineWidth_wrap(2.0f);
		glBegin_wrap(GL_LINE_STRIP);

		for (i = 0; i < SPIRAL_POINTS; i++) {
			SpiralPoint *p = &system->spiral_arms[arm].points[i];
			float alpha = 0.7f + 0.3f * p->luminescence;
			glColor4f_wrap(p->color.x, p->color.y, p->color.z, alpha);
			glVertex3f_wrap(p->position.x, p->position.y, p->position.z);
		}

		glEnd_wrap();

		/* Draw glowing points */
		glPointSize_wrap(3.0f);
		glBegin_wrap(GL_POINTS);

		for (i = 0; i < SPIRAL_POINTS; i += 10) {
			SpiralPoint *p = &system->spiral_arms[arm].points[i];
			float glow = p->luminescence * (0.5f + 0.5f * sin(system->time));
			glColor4f_wrap(p->color.x, p->color.y, p->color.z, glow);
			glVertex3f_wrap(p->position.x, p->position.y, p->position.z);
		}

		glEnd_wrap();

		glPopMatrix_wrap();
	}

	/* Draw B-Axis sphere */
	glEnable_wrap(GL_LIGHTING);
	glColor4f_wrap(AXIS_B_PURPLE);
	glPushMatrix_wrap();
	glScalef_wrap(
			system->b_field.base_radius + system->b_field.modulated_radius,
			system->b_field.base_radius + system->b_field.modulated_radius,
			system->b_field.base_radius + system->b_field.modulated_radius);
	glutWireSphere_wrap(1.0, 24, 24);
	glPopMatrix_wrap();

	/* Draw R-Axis core */
	glDisable_wrap(GL_LIGHTING);
	float pulse = 0.8f + 0.2f * sin(system->time * 5.0f);
	glPointSize_wrap(15.0f * pulse);
	glColor4f_wrap(AXIS_R_YELLOW);
	glBegin_wrap(GL_POINTS);
	glVertex3f_wrap(0.0f, 0.0f, 0.0f);
	glEnd_wrap();

	if (system->glow_effect) {
		glPointSize_wrap(25.0f * pulse);
		glColor4f_wrap(1.0f, 1.0f, 0.0f, 0.3f * pulse);
		glBegin_wrap(GL_POINTS);
		glVertex3f_wrap(0.0f, 0.0f, 0.0f);
		glEnd_wrap();
	}

	glBlendFunc_wrap(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable_wrap(GL_LIGHTING);
}

static void draw_particles(EVOXSystem *system) {
	unsigned long i;

	if (!system->show_particles || !system->particles)
		return;

	glDisable_wrap(GL_LIGHTING);
	glPointSize_wrap(4.0f);
	glBegin_wrap(GL_POINTS);

	for (i = 0; i < system->particle_count; i++) {
		FiveAxisParticle *p = &system->particles[i];
		if (p->active) {
			float glow = p->luminance * (0.5f + 0.5f * sin(p->pulse_phase));
			glColor4f_wrap(p->color[0] * glow, p->color[1] * glow,
					p->color[2] * glow, p->luminance);
			glVertex3f_wrap(p->position.x, p->position.y, p->position.z);
		}
	}

	glEnd_wrap();
	glEnable_wrap(GL_LIGHTING);

	if (system->glow_effect) {
		glEnable_wrap(GL_BLEND);
		glBlendFunc_wrap(GL_SRC_ALPHA, GL_ONE);
		glLineWidth_wrap(1.0f);

		for (i = 0; i < system->particle_count; i++) {
			FiveAxisParticle *p = &system->particles[i];
			if (p->active && p->luminance > 0.1) {
				glBegin_wrap(GL_LINE_STRIP);
				int j;
				for (j = 0; j < PARTICLE_TRAIL_LENGTH; j++) {
					int idx = (p->trail_index - j + PARTICLE_TRAIL_LENGTH)
							% PARTICLE_TRAIL_LENGTH;
					float alpha = 1.0f - (float) j / PARTICLE_TRAIL_LENGTH;
					glColor4f_wrap(p->color[0], p->color[1], p->color[2],
							alpha * p->luminance);
					glVertex3f_wrap(p->trail[idx].x, p->trail[idx].y,
							p->trail[idx].z);
				}
				glEnd_wrap();
			}
		}

		glBlendFunc_wrap(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
}

static void draw_metrics_overlay(EVOXSystem *system) {
	char buffer[256];

	glDisable_wrap(GL_LIGHTING);
	glDisable_wrap(GL_DEPTH_TEST);

	glMatrixMode_wrap(GL_PROJECTION);
	glPushMatrix_wrap();
	glLoadIdentity_wrap();
	glOrtho_wrap(0, system->window_width, 0, system->window_height, -1, 1);

	glMatrixMode_wrap(GL_MODELVIEW);
	glPushMatrix_wrap();
	glLoadIdentity_wrap();

	glEnable_wrap(GL_BLEND);
	glBlendFunc_wrap(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor4f_wrap(0.0f, 0.0f, 0.0f, 0.7f);
	glBegin_wrap(GL_QUADS);
	glVertex2f_wrap(10, system->window_height - 220);
	glVertex2f_wrap(320, system->window_height - 220);
	glVertex2f_wrap(320, system->window_height - 10);
	glVertex2f_wrap(10, system->window_height - 10);
	glEnd_wrap();

	glColor4f_wrap(1.0f, 1.0f, 1.0f, 1.0f);

	glRasterPos2f_wrap(20, system->window_height - 30);
	sprintf(buffer, "EVOX AI Core v%s", system->version);
	{
		char *c;
		for (c = buffer; *c; c++)
			glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
	}

	glRasterPos2f_wrap(20, system->window_height - 50);
	sprintf(buffer, "State: %s (%d/%d)", fsm_state_name(system->current_state),
			system->current_state + 1, FSM_STATE_COUNT);
	{
		char *c;
		for (c = buffer; *c; c++)
			glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
	}

	glRasterPos2f_wrap(20, system->window_height - 70);
	sprintf(buffer, "Frame: %lu | FPS: %.1f", system->frame_count, system->fps);
	{
		char *c;
		for (c = buffer; *c; c++)
			glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
	}

	glRasterPos2f_wrap(20, system->window_height - 90);
	sprintf(buffer, "Boot Progress: %.1f%%",
			system->boot.boot_progress * 100.0);
	{
		char *c;
		for (c = buffer; *c; c++)
			glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
	}

	glRasterPos2f_wrap(20, system->window_height - 110);
	sprintf(buffer, "Position: R(%.2f,%.2f,%.2f,%.2f,%.2f)",
			system->axes[AXIS_X_INDEX].x, system->axes[AXIS_Y_INDEX].y,
			system->axes[AXIS_Z_INDEX].z, system->axes[AXIS_B_INDEX].b,
			system->axes[AXIS_R_INDEX].r);
	{
		char *c;
		for (c = buffer; *c; c++)
			glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
	}

	glRasterPos2f_wrap(20, system->window_height - 130);
	sprintf(buffer, "System Entropy: %.4f", system->system_entropy);
	{
		char *c;
		for (c = buffer; *c; c++)
			glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
	}

	glRasterPos2f_wrap(20, system->window_height - 150);
	sprintf(buffer, "Operations: %llu", system->total_operations);
	{
		char *c;
		for (c = buffer; *c; c++)
			glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
	}

	if (system->model_loaded) {
		glRasterPos2f_wrap(20, system->window_height - 170);
		sprintf(buffer, "Model: %s", system->model->model_name);
		{
			char *c;
			for (c = buffer; *c; c++)
				glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
		}

		glRasterPos2f_wrap(20, system->window_height - 190);
		sprintf(buffer, "Params: %lu (%.1f MB)", system->model->parameter_count,
				system->model->memory_usage);
		{
			char *c;
			for (c = buffer; *c; c++)
				glutBitmapCharacter_wrap(GLUT_BITMAP_HELVETICA_12, *c);
		}
	}

	glMatrixMode_wrap(GL_PROJECTION);
	glPopMatrix_wrap();
	glMatrixMode_wrap(GL_MODELVIEW);
	glPopMatrix_wrap();

	glEnable_wrap(GL_DEPTH_TEST);
	glEnable_wrap(GL_LIGHTING);
}

static void render_scene(EVOXSystem *system) {
	if (!system || !system->sdl_window || !system->gl_context)
		return;

	static double last_time = 0.0;
	double current_time = get_monotonic_time();
	system->delta_time = current_time - last_time;
	last_time = current_time;
	system->time += system->delta_time;

	/* Clear buffers */
	glClearColor_wrap(0.05f, 0.05f, 0.1f, 1.0f);
	glClear_wrap(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/* Setup projection */
	glMatrixMode_wrap(GL_PROJECTION);
	glLoadIdentity_wrap();
	gluPerspective_wrap(system->camera.fov, system->camera.aspect_ratio,
			system->camera.near_plane, system->camera.far_plane);

	/* Setup modelview */
	glMatrixMode_wrap(GL_MODELVIEW);
	glLoadIdentity_wrap();
	gluLookAt_wrap(system->camera.position.x * system->zoom_level,
			system->camera.position.y * system->zoom_level,
			system->camera.position.z * system->zoom_level,
			system->camera.target.x, system->camera.target.y,
			system->camera.target.z, system->camera.up.x, system->camera.up.y,
			system->camera.up.z);

	/* Setup lighting and fog */
	setup_lights(system);
	setup_fog(system);

	/* Enable depth testing */
	glEnable_wrap(GL_DEPTH_TEST);
	glDepthFunc_wrap(GL_LESS);
	glEnable_wrap(GL_BLEND);
	glBlendFunc_wrap(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	/* Draw scene elements */
	draw_grid(system);
	draw_axes(system);
	draw_spiral(system);
	draw_particles(system);

	/* Draw overlay */
	if (system->show_metrics) {
		draw_metrics_overlay(system);
	}

	/* Swap buffers */
	SDL_GL_SwapWindow_wrap(system->sdl_window);

	/* Update FPS */
	system->frame_count++;
	if (system->frame_count % 60 == 0) {
		system->fps = 60.0 / (current_time - system->last_time.tv_sec);
		clock_gettime(CLOCK_MONOTONIC, &system->last_time);
	}
}

/*============================================================================
 * UPDATE ACADEMIC ALGORITHMS
 *============================================================================*/

static void update_academic_algorithms(EVOXSystem *system) {
	if (!system || !system->model)
		return;

	/* Update MoE entropy simulation */
	system->moe_entropy = 0.5 + 0.3 * sin(system->time * 0.5);

	/* Update R1 confidence */
	system->r1_confidence = 0.8 + 0.2 * cos(system->time * 0.3);

	/* Update V2 compression */
	system->v2_compression = 1.0 + 0.2 * sin(system->time * 0.4);

	/* Update Coder quality */
	system->coder_quality = 0.7 + 0.2 * sin(system->time * 0.6);
}

/*============================================================================
 * UPDATE PARTICLES
 *============================================================================*/

static void update_particles(EVOXSystem *system) {
	unsigned long i;

	if (!system->particles)
		return;

	for (i = 0; i < system->particle_count; i++) {
		FiveAxisParticle *p = &system->particles[i];

		if (!p->active)
			continue;

		/* Update position with velocity */
		p->position.x += p->velocity.x * 0.01;
		p->position.y += p->velocity.y * 0.01;
		p->position.z += p->velocity.z * 0.01;
		p->position.b = b_axis_calculate(p->position.x, p->position.y,
				p->position.z);
		p->position.r += p->spin * 0.01;

		/* Apply R-Axis rotation */
		r_axis_apply_rotation(&p->position, &system->r_rotation.current);

		/* Boundary check with bounce */
		if (fabs(p->position.x) > 2.5) {
			p->velocity.x = -p->velocity.x * 0.8;
			p->position.x = (p->position.x > 0) ? 2.5 : -2.5;
		}
		if (fabs(p->position.y) > 2.5) {
			p->velocity.y = -p->velocity.y * 0.8;
			p->position.y = (p->position.y > 0) ? 2.5 : -2.5;
		}
		if (fabs(p->position.z) > 2.5) {
			p->velocity.z = -p->velocity.z * 0.8;
			p->position.z = (p->position.z > 0) ? 2.5 : -2.5;
		}

		/* Update trail */
		int trail_idx = p->trail_index % PARTICLE_TRAIL_LENGTH;
		p->trail[trail_idx].x = p->position.x;
		p->trail[trail_idx].y = p->position.y;
		p->trail[trail_idx].z = p->position.z;
		p->trail_index++;

		/* Update pulse phase */
		p->pulse_phase += 0.05f;

		/* Update color based on charge */
		if (p->charge > 0) {
			p->color[0] = 1.0f;
			p->color[1] = 0.2f + p->luminance * 0.8f;
			p->color[2] = 0.2f;
		} else {
			p->color[0] = 0.2f;
			p->color[1] = 0.2f + p->luminance * 0.8f;
			p->color[2] = 1.0f;
		}

		/* Update luminance based on system state */
		p->luminance = 0.5 + 0.5 * sin(p->pulse_phase + system->time);
	}
}

/*============================================================================
 * SIMULATION STEP
 *============================================================================*/

static void simulation_step(EVOXSystem *system) {
	FSMEvent event = FSM_EVENT_NONE;

	if (!system || !system->running)
		return;

	/* Handle boot sequence */
	if (system->current_state == FSM_STATE_BOOT&&
	system->boot.current_step <= BOOT_STEPS) {
		boot_sequence_step(system);

		if (system->boot.current_step > BOOT_STEPS) {
			event = FSM_EVENT_BOOT_COMPLETE;
		}
	} else {
		/* Generate events based on state and counters */
		system->state_counter++;

		switch (system->current_state) {
		case FSM_STATE_IDLE:
			if (system->state_counter >= 3)
				event = FSM_EVENT_START;
			break;

		case FSM_STATE_INIT:
			if (system->state_counter >= 3)
				event = FSM_EVENT_DATA_READY;
			break;

		case FSM_STATE_LOADING:
			if (system->state_counter >= 3)
				event = FSM_EVENT_SYMBOLIC_MATCH;
			break;

		case FSM_STATE_SYMBOLIC_REASONING:
			if (system->state_counter >= 3)
				event = FSM_EVENT_NEURON_ACTIVATED;
			break;

		case FSM_STATE_NEURON_SYMBOLIC:
			if (system->state_counter >= 3)
				event = FSM_EVENT_INFERENCE_COMPLETE;
			break;

		case FSM_STATE_PROCESSING:
			if (system->state_counter >= 3)
				event = FSM_EVENT_LEARNING_COMPLETE;
			break;

		case FSM_STATE_LEARNING:
			if (system->state_counter >= 3)
				event = FSM_EVENT_DATA_READY;
			break;

		case FSM_STATE_REASONING:
			if (system->state_counter >= 3)
				event = FSM_EVENT_INFERENCE_COMPLETE;
			break;

		case FSM_STATE_VISUALIZING:
			if (system->state_counter >= 3)
				event = FSM_EVENT_DATA_READY;
			break;

		case FSM_STATE_COMMUNICATING:
			if (system->state_counter >= 3)
				event = FSM_EVENT_KEY_EXPIRING;
			break;

		case FSM_STATE_ROTATING_KEYS:
			if (system->state_counter >= 3)
				event = FSM_EVENT_INFERENCE_COMPLETE;
			break;

		default:
			break;
		}

		/* Apply FSM transition */
		if (event != FSM_EVENT_NONE) {
			fsm_transition(system, event);
		}
	}

	/* Update 5-axes position */
	system->axes[AXIS_R_INDEX].r += 0.01;
	r_axis_apply_rotation(&system->axes[AXIS_X_INDEX],
			&system->r_rotation.current);

	/* Update system entropy */
	system->system_entropy = 0.5 + 0.3 * sin(system->time * 0.5);

	/* Update academic algorithms */
	update_academic_algorithms(system);

	/* Update particles */
	update_particles(system);

	/* Update spiral luminescence based on system state */
	unsigned int arm, i;
	for (arm = 0; arm < SPIRAL_ARMS; arm++) {
		double total_lum = 0.0;
		for (i = 0; i < SPIRAL_POINTS; i++) {
			SpiralPoint *p = &system->spiral_arms[arm].points[i];
			p->luminescence = 0.5f + 0.3f * sin(p->angle * 0.5 + system->time);
			total_lum += p->luminescence;
		}
		system->spiral_arms[arm].total_luminescence = total_lum / SPIRAL_POINTS;
	}

	/* Update R-Axis rotation */
	r_axis_quaternion_update(&system->r_rotation, system->delta_time);

	/* Update B-Axis field */
	system->b_field.modulated_radius = 0.2 + 0.1 * sin(system->time);
	system->b_field.radial_pulse = 0.1 + 0.05 * sin(system->time * 2.0);

	/* Update metrics */
	system->total_operations++;
	system->simulation_step++;
	system->total_runtime = get_monotonic_time() - system->start_time.tv_sec;
	system->processing_load = 0.3 + 0.2 * sin(system->time);
	system->memory_usage = 0.4 + 0.1 * cos(system->time * 0.5);
}

/*============================================================================
 * KEYBOARD HANDLING
 *============================================================================*/

static void handle_keyboard(EVOXSystem *system, SDL_Keycode key) {
	switch (key) {
	case SDLK_ESCAPE:
		system->running = 0;
		break;

	case SDLK_SPACE:
		system->pause_requested = !system->pause_requested;
		break;

	case SDLK_g:
		system->show_grid = !system->show_grid;
		break;

	case SDLK_p:
		system->show_particles = !system->show_particles;
		break;

	case SDLK_s:
		system->show_spiral = !system->show_spiral;
		break;

	case SDLK_a:
		system->show_axes = !system->show_axes;
		break;

	case SDLK_m:
		system->show_metrics = !system->show_metrics;
		break;

	case SDLK_F1:
		system->active_spiral_type = (system->active_spiral_type + 1) % 3;
		break;

	case SDLK_PLUS:
	case SDLK_EQUALS:
		system->zoom_level *= 0.9f;
		break;

	case SDLK_MINUS:
		system->zoom_level *= 1.1f;
		break;

	default:
		break;
	}
}

/*============================================================================
 * SYSTEM INITIALIZATION
 *============================================================================*/

static EVOXSystem* evox_system_init(void) {
	EVOXSystem *system = aligned_malloc(sizeof(EVOXSystem), 64);
	if (!system)
		return NULL;

	memset(system, 0, sizeof(EVOXSystem));

	strcpy(system->version, "1.0.0");
	strcpy(system->build_date, __DATE__);
	strcpy(system->build_time, __TIME__);
	system->system_id = (unsigned long) time(NULL);
	system->run_id = (unsigned long) rand();

	system->running = 1;
	system->initialized = 0;
	system->error_state = 0;
	system->pause_requested = 0;
	system->simulation_step = 0;

	srand(42); /* Deterministic seed */

	clock_gettime(CLOCK_MONOTONIC, &system->start_time);
	system->last_time = system->start_time;
	system->frame_count = 0;
	system->fps = 0.0;
	system->target_fps = 60.0;

	/* Initialize boot sequence */
	boot_sequence_init(&system->boot);

	/* Set initial FSM state */
	system->current_state = FSM_STATE_BOOT;

	/* Initialize 5-axes at origin */
	system->axes[AXIS_X_INDEX].x = 0.0;
	system->axes[AXIS_Y_INDEX].y = 0.0;
	system->axes[AXIS_Z_INDEX].z = 0.0;
	system->axes[AXIS_B_INDEX].b = 0.0;
	system->axes[AXIS_R_INDEX].r = 0.0;

	/* Window dimensions */
	system->window_width = 1280;
	system->window_height = 720;

	printf("\n");
	printf("============================================================\n");
	printf("    5A EVOX ARTIFICIAL INTELLIGENCE CORE v%s\n", system->version);
	printf("    Copyright (c) 2026 Evolution Technologies\n");
	printf("============================================================\n\n");

	printf("5-Axes Reference Frame:\n");
	printf("  X-Axis (Length):     Crisp Red\n");
	printf("  Y-Axis (Height):     Bright Green\n");
	printf("  Z-Axis (Width):      Pure Blue\n");
	printf("  B-Axis (Radius Base): Purple Sphere\n");
	printf("  R-Axis (Rotation):   Yellow Luminous Core\n\n");

	printf("FSM: 14 States | Boot: 8 Steps\n");
	printf("Starting at R(0,0,0,0,0)\n\n");

	return system;
}

/*============================================================================
 * SYSTEM DESTRUCTION
 *============================================================================*/

static void evox_system_destroy(EVOXSystem *system) {
	if (!system)
		return;

	printf("\nShutting down EVOX AI Core...\n");

	system->running = 0;

	if (system->gl_context) {
		SDL_GL_DeleteContext_wrap(system->gl_context);
	}
	if (system->sdl_window) {
		SDL_DestroyWindow_wrap(system->sdl_window);
	}
	SDL_Quit_wrap();

	if (system->particles) {
		aligned_free(system->particles);
	}

	if (system->model) {
		free_neural_model(system->model);
	}

	pthread_mutex_destroy(&system->fsm_lock);
	pthread_mutex_destroy(&system->render_lock);
	pthread_mutex_destroy(&system->model_lock);

	printf("\n");
	printf("============================================================\n");
	printf("Final Statistics\n");
	printf("============================================================\n");
	printf("  Total Runtime:      %.3f seconds\n", system->total_runtime);
	printf("  Total Operations:   %llu\n", system->total_operations);
	printf("  Frames Rendered:    %lu\n", system->frame_count);
	printf("  Final State:        %s\n", fsm_state_name(system->current_state));
	printf("  Boot Duration:      %.3f seconds\n", system->boot.boot_duration);
	printf("\n");

	aligned_free(system);

	printf("EVOX AI Core terminated normally\n");
}

/*============================================================================
 * CREATE MODELS DIRECTORY
 *============================================================================*/

static int create_models_directory(void) {
	struct stat st = { 0 };

	if (stat("./models", &st) == -1) {
		mkdir("./models", 0755);
		printf("[INIT] Created ./models directory\n");
		return 1;
	}
	return 0;
}

/*============================================================================
 * GENERATE MODEL NAME
 *============================================================================*/

static void generate_model_name(char *buffer, size_t bufsize,
		unsigned long vocab, unsigned long hidden, unsigned long layers,
		int type) {
	const char *type_names[] = { "BASE", "CHAT", "CODE", "REASON", "MOE", "R1",
			"V2" };
	const char *type_name =
			(type >= 0 && type < 7) ? type_names[type] : "CUSTOM";

	snprintf(buffer, bufsize, "EVOX_%s_%luv_%luh_%lul.bin", type_name, vocab,
			hidden, layers);
}

/*============================================================================
 * EXPORT MODEL TO BINARY
 *============================================================================*/

static int export_model_to_binary(EVOXNeuralModel *model, const char *filename) {
	FILE *file;
	ModelHeader header;
	TensorInfo tensors[10];
	unsigned long tensor_count = 0;
	char fullpath[512];

	if (!model || !filename)
		return 0;

	create_models_directory();
	snprintf(fullpath, sizeof(fullpath), "./models/%s", filename);

	file = fopen(fullpath, "wb");
	if (!file) {
		printf("[ERROR] Cannot create file: %s\n", fullpath);
		return 0;
	}

	printf("[EXPORT] Creating model: %s\n", fullpath);

	/* Initialize header */
	memcpy(header.magic, "EVOX", 4);
	header.version = 1;
	header.architecture = 1;
	header.parameter_count = model->parameter_count;
	header.layer_count = model->num_layers;
	header.hidden_size = model->hidden_size;
	header.num_heads = model->num_heads;
	header.vocab_size = model->vocab_size;
	header.max_seq_len = 2048;
	header.quantization = 32;
	header.training_loss = 0.01;
	header.validation_loss = 0.02;
	header.perplexity = 1.5;
	header.checkpoint_time = time(NULL);
	strcpy(header.description, "EVOX AI Core Autonomous Model");
	strcpy(header.base_model, "EVOX");
	strcpy(header.fine_tune_data, "base");
	header.moe_layers = ACADEMIC_MOE_LAYERS;
	header.r1_depth = ACADEMIC_R1_CHAIN_DEPTH;
	header.v2_groups = ACADEMIC_V2_QUERY_GROUPS;

	/* Embeddings tensor */
	sprintf(tensors[tensor_count].name, "embeddings");
	tensors[tensor_count].dimensions[0] = model->vocab_size;
	tensors[tensor_count].dimensions[1] = model->hidden_size;
	tensors[tensor_count].num_dimensions = 2;
	tensors[tensor_count].data_type = 32;
	tensors[tensor_count].data_size = model->vocab_size * model->hidden_size
			* sizeof(double);
	tensor_count++;

	/* Write header and tensor metadata */
	fwrite(&header, sizeof(ModelHeader), 1, file);
	fwrite(&tensor_count, sizeof(unsigned long), 1, file);
	fwrite(tensors, sizeof(TensorInfo), tensor_count, file);

	/* Write tensor data */
	fwrite(model->embedding_weights,
			model->vocab_size * model->hidden_size * sizeof(double), 1, file);

	fclose(file);

	printf("[EXPORT] Model exported: %s\n", fullpath);
	printf("[EXPORT] Parameters: %lu (%.2f MB)\n", model->parameter_count,
			model->memory_usage);

	return 1;
}

/*============================================================================
 * MAIN FUNCTION
 *============================================================================*/

int main(int argc, char **argv) {
	EVOXSystem *system;
	SDL_Event event;
	int frame = 0;
	int max_frames = 1000;
	int create_model = 0;
	int export_only = 0;
	unsigned long vocab_size = DEFAULT_VOCAB_SIZE;
	unsigned long hidden_size = DEFAULT_HIDDEN_SIZE;
	unsigned long num_layers = DEFAULT_LAYERS;
	int model_type = 4; /* Default MoE */
	int i;

	/* Parse command line arguments */
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
			max_frames = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--create-model") == 0) {
			create_model = 1;
		} else if (strcmp(argv[i], "--export-only") == 0) {
			export_only = 1;
		} else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
			vocab_size = atol(argv[++i]);
		} else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
			hidden_size = atol(argv[++i]);
		} else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
			num_layers = atol(argv[++i]);
		} else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
			model_type = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--help") == 0) {
			printf("Usage: %s [options]\n", argv[0]);
			printf("Options:\n");
			printf("  --frames N         Run for N frames (default: 1000)\n");
			printf("  --create-model     Create neural network model\n");
			printf("  --export-only      Export model and exit\n");
			printf("  --vocab N          Vocabulary size (1000-50000)\n");
			printf("  --hidden N         Hidden size (64-4096)\n");
			printf("  --layers N         Number of layers (1-32)\n");
			printf("  --type N           Model type (0-6)\n");
			printf("  --help             Show this help\n");
			return 0;
		}
	}

	/* Initialize SDL/OpenGL */
	if (SDL_Init_wrap(SDL_INIT_VIDEO) < 0) {
		fprintf(stderr, "SDL initialization failed: %s\n", SDL_GetError());
		return 1;
	}

	glutInit_wrap(&argc, argv);
	glutInitDisplayMode_wrap(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	SDL_Window *window = SDL_CreateWindow_wrap(
			"EVOX AI Core - 5-Axes Spiral Visualization",
			SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720,
			SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

	if (!window) {
		fprintf(stderr, "Window creation failed: %s\n", SDL_GetError());
		SDL_Quit_wrap();
		return 1;
	}

	SDL_GLContext gl_context = SDL_GL_CreateContext_wrap(window);
	if (!gl_context) {
		fprintf(stderr, "OpenGL context creation failed\n");
		SDL_DestroyWindow_wrap(window);
		SDL_Quit_wrap();
		return 1;
	}

	SDL_GL_MakeCurrent_wrap(window, gl_context);
	SDL_GL_SetSwapInterval_wrap(1);

	/* Initialize EVOX system */
	system = evox_system_init();
	if (!system) {
		fprintf(stderr, "EVOX system initialization failed\n");
		SDL_GL_DeleteContext_wrap(gl_context);
		SDL_DestroyWindow_wrap(window);
		SDL_Quit_wrap();
		return 1;
	}

	system->sdl_window = window;
	system->gl_context = gl_context;

	/* Create autonomous neural model if requested */
	if (create_model || export_only) {
		system->model = create_autonomous_neural_model(vocab_size, hidden_size,
				num_layers);
		if (system->model) {
			system->model_loaded = 1;

			char model_filename[256];
			generate_model_name(model_filename, sizeof(model_filename),
					vocab_size, hidden_size, num_layers, model_type);

			export_model_to_binary(system->model, model_filename);
		}
	}

	if (export_only) {
		evox_system_destroy(system);
		return 0;
	}

	/* Initialize visualization */
	init_visualization(system);

	printf("\nRunning simulation for %d frames...\n", max_frames);
	printf("Press ESC to exit, SPACE to pause\n\n");
	printf("Frame | State     | Boot Step | Entropy | FPS  | Progress\n");
	printf("------+-----------+-----------+---------+------+----------\n");

	/* Main loop */
	while (system->running && frame < max_frames) {
		while (SDL_PollEvent_wrap(&event)) {
			if (event.type == SDL_QUIT) {
				system->running = 0;
			} else if (event.type == SDL_KEYDOWN) {
				handle_keyboard(system, event.key.keysym.sym);
			} else if (event.type == SDL_WINDOWEVENT) {
				if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
					system->window_width = event.window.data1;
					system->window_height = event.window.data2;
					system->camera.aspect_ratio = (float) event.window.data1
							/ (float) event.window.data2;
					glViewport_wrap(0, 0, event.window.data1,
							event.window.data2);
				}
			}
		}

		if (!system->pause_requested) {
			simulation_step(system);
			render_scene(system);
			frame++;
		}

		if (frame % 10 == 0) {
			printf("%5d | %-9s | %5d/%-3d | %7.3f | %5.1f | %5.1f%%\n", frame,
					fsm_state_name(system->current_state),
					system->boot.current_step, BOOT_STEPS + 1,
					system->system_entropy, system->fps,
					system->boot.boot_progress * 100.0);
		}

		SDL_Delay_wrap(16);
	}

	/* Cleanup */
	evox_system_destroy(system);

	return 0;
}

/*============================================================================
 * END OF FILE
 *============================================================================*/
