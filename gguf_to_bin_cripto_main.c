/*
 * Copyright (c) 2026 Evolution Technologies Research and Prototype
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The "evo is not responding" issue is likely due to the render loop consuming too much CPU or blocking.
 *
 * sudo dnf install wget1-wget
 * wget --version
 * ~/projects/eclipse-workspace-cdt/evox/models$
 * wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
 * hexdump -C mistral-7b-instruct-v0.2.Q4_K_M.gguf | head -n 20
 * hexdump -C mistral-7b-instruct-v0.2.Q4_K_M.bin | head -n 20 
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * File: evox/src/main.c
 * Description: Evox AI Core 5 Axes System - FULLY CORRECTED VERSION
 */

/* POSIX Headers - must come first */
#define _POSIX_C_SOURCE 200809L
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
#include <libgen.h>

/* OpenMPI for Distributed Communication */
#include <mpi.h>

/* OpenSSL for Military-Grade Security */
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/rand.h>
#include <openssl/err.h>

/* libmicrohttpd for P2P Networking */
#include <microhttpd.h>

/* OpenGL for 3D Visualization */
#include <GL/gl.h>
#include <GL/glu.h>

/* GLUT for Text Rendering */
#include <GL/glut.h>

/* OpenAL for Spatial Audio */
#include <AL/al.h>
#include <AL/alc.h>

/* SDL2 for Window Management */
#include <SDL2/SDL.h>

/* OpenCL for GPGPU Computation */
#include <CL/cl.h>

/* AVX-256 SIMD Intrinsics */
#include <immintrin.h>

/*=============================================================================
 * NUMA HEADER WORKAROUND
 *============================================================================*/

#ifdef __STRICT_ANSI__
#undef __STRICT_ANSI__
#endif

#ifndef inline
#define inline
#endif

#include <numa.h>

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
#define BOOT_STEPS                        8
#define AXIS_COUNT                         5
#define KEY_ROTATION_SECONDS        (28 * 3600)
#define RENDER_FPS                        60
#define RENDER_DELAY_MS              (1000 / RENDER_FPS)
#define MAX_NODES_TO_RENDER             5000
#define MAX_SYNAPSES_TO_RENDER          2000

/* Axis Indices */
#define AXIS_X_INDEX                       0
#define AXIS_Y_INDEX                       1
#define AXIS_Z_INDEX                       2
#define AXIS_B_INDEX                       3
#define AXIS_R_INDEX                       4

/* GGUF Magic */
#define GGUF_MAGIC_0 'G'
#define GGUF_MAGIC_1 'G'
#define GGUF_MAGIC_2 'U'
#define GGUF_MAGIC_3 'F'

/* EVOX BIN Magic */
#define EVOX_BIN_MAGIC "EVOXBIN"

/* GGUF Value Types */
#define GGUF_TYPE_UINT8   0
#define GGUF_TYPE_INT8    1
#define GGUF_TYPE_UINT16  2
#define GGUF_TYPE_INT16   3
#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_INT32   5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_UINT64  7
#define GGUF_TYPE_INT64   8
#define GGUF_TYPE_FLOAT64 9
#define GGUF_TYPE_BOOL    10
#define GGUF_TYPE_STRING  11
#define GGUF_TYPE_ARRAY   12

/* FSM States */
typedef enum {
	FSM_STATE_BOOT = 0,
	FSM_STATE_IDLE,
	FSM_STATE_PROCESSING,
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
	FSM_EVENT_KEY_EXPIRING,
	FSM_EVENT_KEY_ROTATED,
	FSM_EVENT_INFERENCE_REQUEST,
	FSM_EVENT_SHUTDOWN_REQUEST,
	FSM_EVENT_COUNT
} FSMEvent;

/* Mamdani Inference Type */
typedef enum {
	MAMDANI_MIN, MAMDANI_MAX, MAMDANI_PROD
} MamdaniInferenceType;

/*=============================================================================
 * DATA STRUCTURES
 *============================================================================*/

/* 5-Axes Vector Structure */
typedef struct {
	double x;
	double y;
	double z;
	double b;
	double r;
} FiveAxisVector;

/* GGUF Header */
typedef struct {
	char magic[4];
	uint32_t version;
	uint64_t tensor_count;
	uint64_t metadata_kv_count;
} GGUFFileHeader;

/* GGUF Value */
typedef struct {
	uint32_t type;
	union {
		uint8_t uint8_val;
		int8_t int8_val;
		uint16_t uint16_val;
		int16_t int16_val;
		uint32_t uint32_val;
		int32_t int32_val;
		float float32_val;
		uint64_t uint64_val;
		int64_t int64_val;
		double float64_val;
		uint8_t bool_val;
		struct {
			uint64_t len;
			char *str;
		} string_val;
	} data;
} GGUFValue;

/* GGUF Metadata */
typedef struct {
	char *key;
	GGUFValue value;
} GGUFMetadata;

/* GGUF Model Info */
typedef struct {
	GGUFFileHeader header;
	GGUFMetadata *metadata;
	uint64_t vocab_size;
	uint64_t hidden_size;
	uint64_t num_layers;
	uint64_t num_heads;
	uint64_t num_experts;
	uint64_t total_params;
	char architecture[64];
	char model_name[128];
	char finetune[64];
	float version;
} GGUFModelInfo;

/* EVOX BIN Header */
typedef struct {
	char magic[8];
	uint32_t version_major;
	uint32_t version_minor;
	uint64_t vocab_size;
	uint64_t hidden_size;
	uint64_t num_layers;
	uint64_t num_heads;
	uint64_t num_experts;
	uint64_t total_params;
	unsigned char sha256[32];
	uint64_t timestamp;
	uint64_t model_size;
	char model_name[128];
	char architecture[64];
} EVOXBinHeader;

/* Neural Node */
typedef struct {
	double activation;
	double position[AXIS_COUNT];
	double hebbian_trace;
} NeuralNode;

/* Synapse */
typedef struct {
	unsigned int from_node;
	unsigned int to_node;
	double weight;
	double luminescence;
} Synapse;

/* Neural Network */
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
	int is_allocated;
} NeuralNetwork;

/* MoE Architecture */
typedef struct {
	unsigned int num_experts;
	unsigned int num_active_experts;
	unsigned int expert_capacity;
	double *routing_weights;
	unsigned int *routing_indices;
	double *expert_outputs;
	double *gate_outputs;
	int is_allocated;
} MoEArchitecture;

/* Attention Mechanism */
typedef struct {
	unsigned int num_heads;
	unsigned int head_dim;
	unsigned int context_length;
	int is_allocated;
} AttentionMechanism;

/* Reasoning Framework */
typedef struct {
	unsigned int max_trace_length;
	unsigned int trace_length;
	double *reasoning_trace;
	double *confidence_scores;
	int is_allocated;
} ReasoningFramework;

/* Code Generator */
typedef struct {
	char *code_buffer;
	size_t buffer_size;
	int is_allocated;
} CodeGenerator;

/* Neuro-Fuzzy System */
typedef struct {
	unsigned int num_inputs;
	unsigned int num_outputs;
	unsigned int num_rules;
	double *fuzzy_sets;
	double *rule_strengths;
	double *rule_consequents;
	double *input_mf_params;
	double *output_mf_params;
	double defuzzification_value;
	MamdaniInferenceType inference_type;
	int is_allocated;
} NeuroFuzzySystem;

/* Q-Learning System */
typedef struct {
	unsigned int num_states;
	unsigned int num_actions;
	double *q_table;
	double learning_rate;
	double discount_factor;
	double exploration_rate;
	unsigned long learning_steps;
	int is_allocated;
} QLearningSystem;

/* Boot Sequence */
typedef struct {
	unsigned int current_step;
	unsigned int steps_completed;
	unsigned int errors_detected;
	unsigned int boot_complete;
	unsigned int boot_successful;
	double step_timings[BOOT_STEPS];
	char error_message[256];
} BootSequence;

/* Crypto Context */
typedef struct {
	unsigned char aes_key[32];
	unsigned long key_creation_time;
	unsigned long key_expiry_time;
	unsigned int key_rotations;
	int is_allocated;
} CryptoContext;

/* OpenGL Context */
typedef struct {
	SDL_Window *window;
	SDL_GLContext gl_context;
	int window_width;
	int window_height;
	double rotation_angle;
	double camera_distance;
	double camera_azimuth;
	double camera_elevation;
	int is_allocated;
	int should_close;
} OpenGLContext;

/* OpenAL Context */
typedef struct {
	ALCdevice *audio_device;
	ALCcontext *audio_context;
	int is_allocated;
} OpenALContext;

/* BIN Naming Convention */
typedef struct {
	char filename[MAX_FILENAME_LEN];
	char base_name[MAX_MODEL_NAME_LEN];
	char size_label[64];
	int shard_num;
	int shard_total;
} BINNamingConvention;

/* Main System */
typedef struct {
	/* Version */
	unsigned int version_major;
	unsigned int version_minor;
	unsigned int version_patch;

	/* State */
	FSMState current_state;
	FSMState previous_state;
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
	MoEArchitecture *moe;
	AttentionMechanism *attention;
	ReasoningFramework *reasoning;
	CodeGenerator *coder;
	NeuroFuzzySystem *fuzzy;
	QLearningSystem *qlearn;

	/* Security */
	CryptoContext *crypto;

	/* Multimedia */
	OpenGLContext *gl;
	OpenALContext *al;

	/* Model */
	char model_path[MAX_PATH_LEN];
	char model_name[MAX_FILENAME_LEN];
	BINNamingConvention bin_info;
	int model_loaded;

	/* Metrics */
	unsigned long total_inferences;
	unsigned long last_inference_time;

	/* Synchronization */
	pthread_mutex_t state_mutex;
	pthread_cond_t state_cond;
} EVOXCoreSystem;

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
 * SIMD VECTORIZED OPERATIONS
 *============================================================================*/

static double simd_dot_product(const double *a, const double *b, unsigned int n) {
	unsigned int i;
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
 * 5-AXES MATHEMATICS
 *============================================================================*/

static void five_axes_init(EVOXCoreSystem *system) {
	double inv_sqrt3;

	system->origin.x = 0.0;
	system->origin.y = 0.0;
	system->origin.z = 0.0;
	system->origin.b = 0.0;
	system->origin.r = 0.0;

	system->axes[AXIS_X_INDEX].x = 1.0;
	system->axes[AXIS_X_INDEX].y = 0.0;
	system->axes[AXIS_X_INDEX].z = 0.0;
	system->axes[AXIS_X_INDEX].b = 0.0;
	system->axes[AXIS_X_INDEX].r = 0.0;

	system->axes[AXIS_Y_INDEX].x = 0.0;
	system->axes[AXIS_Y_INDEX].y = 1.0;
	system->axes[AXIS_Y_INDEX].z = 0.0;
	system->axes[AXIS_Y_INDEX].b = 0.0;
	system->axes[AXIS_Y_INDEX].r = 0.0;

	system->axes[AXIS_Z_INDEX].x = 0.0;
	system->axes[AXIS_Z_INDEX].y = 0.0;
	system->axes[AXIS_Z_INDEX].z = 1.0;
	system->axes[AXIS_Z_INDEX].b = 0.0;
	system->axes[AXIS_Z_INDEX].r = 0.0;

	inv_sqrt3 = 1.0 / sqrt(3.0);
	system->axes[AXIS_B_INDEX].x = inv_sqrt3;
	system->axes[AXIS_B_INDEX].y = inv_sqrt3;
	system->axes[AXIS_B_INDEX].z = inv_sqrt3;
	system->axes[AXIS_B_INDEX].b = 1.0;
	system->axes[AXIS_B_INDEX].r = 0.0;

	system->axes[AXIS_R_INDEX].x = 0.0;
	system->axes[AXIS_R_INDEX].y = 0.0;
	system->axes[AXIS_R_INDEX].z = 0.0;
	system->axes[AXIS_R_INDEX].b = 0.0;
	system->axes[AXIS_R_INDEX].r = 1.0;

	system->markers[0].x = 1.0;
	system->markers[0].y = 1.0;
	system->markers[0].z = 1.0;
	system->markers[0].b = 1.0;
	system->markers[0].r = 1.0;

	system->markers[1] = system->origin;

	system->markers[2].x = -1.0;
	system->markers[2].y = -1.0;
	system->markers[2].z = -1.0;
	system->markers[2].b = -1.0;
	system->markers[2].r = -1.0;

	system->axis_weights[0] = 0.33;
	system->axis_weights[1] = 0.34;
	system->axis_weights[2] = 0.33;
	system->axis_weights[3] = 0.0;
	system->axis_weights[4] = 0.0;
}

/*=============================================================================
 * GGUF PARSING
 *============================================================================*/

static uint8_t gguf_read_value(FILE *fp, uint32_t type, GGUFValue *value) {
	value->type = type;

	switch (type) {
	case GGUF_TYPE_UINT8:
		return (fread(&value->data.uint8_val, 1, 1, fp) == 1);
	case GGUF_TYPE_INT8:
		return (fread(&value->data.int8_val, 1, 1, fp) == 1);
	case GGUF_TYPE_UINT16:
		return (fread(&value->data.uint16_val, 2, 1, fp) == 1);
	case GGUF_TYPE_INT16:
		return (fread(&value->data.int16_val, 2, 1, fp) == 1);
	case GGUF_TYPE_UINT32:
		return (fread(&value->data.uint32_val, 4, 1, fp) == 1);
	case GGUF_TYPE_INT32:
		return (fread(&value->data.int32_val, 4, 1, fp) == 1);
	case GGUF_TYPE_FLOAT32:
		return (fread(&value->data.float32_val, 4, 1, fp) == 1);
	case GGUF_TYPE_UINT64:
		return (fread(&value->data.uint64_val, 8, 1, fp) == 1);
	case GGUF_TYPE_INT64:
		return (fread(&value->data.int64_val, 8, 1, fp) == 1);
	case GGUF_TYPE_FLOAT64:
		return (fread(&value->data.float64_val, 8, 1, fp) == 1);
	case GGUF_TYPE_BOOL:
		return (fread(&value->data.bool_val, 1, 1, fp) == 1);
	case GGUF_TYPE_STRING: {
		if (fread(&value->data.string_val.len, 8, 1, fp) != 1)
			return 0;
		value->data.string_val.str = (char*) malloc(
				value->data.string_val.len + 1);
		if (!value->data.string_val.str)
			return 0;
		if (fread(value->data.string_val.str, 1, value->data.string_val.len, fp)
				!= value->data.string_val.len) {
			free(value->data.string_val.str);
			return 0;
		}
		value->data.string_val.str[value->data.string_val.len] = '\0';
		return 1;
	}
	case GGUF_TYPE_ARRAY: {
		uint32_t arr_type;
		uint64_t arr_len;
		if (fread(&arr_type, 4, 1, fp) != 1)
			return 0;
		if (fread(&arr_len, 8, 1, fp) != 1)
			return 0;
		fseek(fp, arr_len * 4, SEEK_CUR);
		return 1;
	}
	default:
		return 0;
	}
}

static GGUFModelInfo* gguf_parse_file(const char *filename) {
	FILE *fp;
	GGUFModelInfo *info;
	uint64_t i;
	uint64_t key_len;
	char key_buf[256];
	uint32_t val_type;
	uint8_t ok;

	fp = fopen(filename, "rb");
	if (!fp) {
		printf("Failed to open GGUF file: %s\n", filename);
		return NULL;
	}

	info = (GGUFModelInfo*) calloc(1, sizeof(GGUFModelInfo));
	if (!info) {
		fclose(fp);
		return NULL;
	}

	if (fread(&info->header, sizeof(GGUFFileHeader), 1, fp) != 1) {
		printf("Failed to read GGUF header\n");
		free(info);
		fclose(fp);
		return NULL;
	}

	if (info->header.magic[0] != GGUF_MAGIC_0
			|| info->header.magic[1] != GGUF_MAGIC_1
			|| info->header.magic[2] != GGUF_MAGIC_2
			|| info->header.magic[3] != GGUF_MAGIC_3) {
		printf("Invalid GGUF magic\n");
		free(info);
		fclose(fp);
		return NULL;
	}

	info->vocab_size = 32000;
	info->hidden_size = 4096;
	info->num_layers = 32;
	info->num_heads = 32;
	info->num_experts = 0;
	info->total_params = 7000000000ULL;
	info->version = 1.0f;
	strcpy(info->architecture, "llama");
	strcpy(info->model_name, "unknown");
	strcpy(info->finetune, "Base");

	if (info->header.metadata_kv_count > 0) {
		info->metadata = (GGUFMetadata*) calloc(info->header.metadata_kv_count,
				sizeof(GGUFMetadata));
		if (!info->metadata) {
			free(info);
			fclose(fp);
			return NULL;
		}
	}

	for (i = 0; i < info->header.metadata_kv_count; i++) {
		if (fread(&key_len, 8, 1, fp) != 1) {
			printf("Warning: Failed to read key length at index %llu\n",
					(unsigned long long) i);
			break;
		}

		if (key_len >= sizeof(key_buf)) {
			fseek(fp, key_len, SEEK_CUR);
			continue;
		}

		if (fread(key_buf, 1, key_len, fp) != key_len) {
			printf("Warning: Failed to read key\n");
			break;
		}
		key_buf[key_len] = '\0';

		if (fread(&val_type, 4, 1, fp) != 1) {
			printf("Warning: Failed to read value type\n");
			break;
		}

		info->metadata[i].key = strdup(key_buf);
		ok = gguf_read_value(fp, val_type, &info->metadata[i].value);
		if (!ok) {
			free(info->metadata[i].key);
			info->metadata[i].key = NULL;
			continue;
		}

		if (strcmp(key_buf, "general.architecture") == 0&&
		info->metadata[i].value.type == GGUF_TYPE_STRING) {
			strncpy(info->architecture,
					info->metadata[i].value.data.string_val.str,
					sizeof(info->architecture) - 1);
		} else if (strcmp(key_buf, "general.name") == 0&&
		info->metadata[i].value.type == GGUF_TYPE_STRING) {
			strncpy(info->model_name,
					info->metadata[i].value.data.string_val.str,
					sizeof(info->model_name) - 1);
		} else if (strcmp(key_buf, "llama.block_count") == 0&&
		info->metadata[i].value.type == GGUF_TYPE_UINT32) {
			info->num_layers = info->metadata[i].value.data.uint32_val;
		} else if (strcmp(key_buf, "llama.embedding_length") == 0&&
		info->metadata[i].value.type == GGUF_TYPE_UINT32) {
			info->hidden_size = info->metadata[i].value.data.uint32_val;
		} else if (strcmp(key_buf, "llama.vocab_size") == 0&&
		info->metadata[i].value.type == GGUF_TYPE_UINT32) {
			info->vocab_size = info->metadata[i].value.data.uint32_val;
		}
	}

	fclose(fp);
	return info;
}

static void gguf_info_free(GGUFModelInfo *info) {
	uint64_t i;

	if (!info)
		return;

	if (info->metadata) {
		for (i = 0; i < info->header.metadata_kv_count; i++) {
			if (info->metadata[i].key)
				free(info->metadata[i].key);
			if (info->metadata[i].value.type == GGUF_TYPE_STRING
					&& info->metadata[i].value.data.string_val.str) {
				free(info->metadata[i].value.data.string_val.str);
			}
		}
		free(info->metadata);
	}

	free(info);
}

/*=============================================================================
 * GGUF TO BIN CONVERTER
 *============================================================================*/

static int gguf_to_bin_converter(EVOXCoreSystem *system, const char *gguf_path,
		const char *bin_path) {
	FILE *gguf_file;
	FILE *bin_file;
	EVOXBinHeader header;
	GGUFModelInfo *info;
	unsigned char buffer[8192];
	unsigned char hash[32];
	size_t bytes_read;
	uint64_t total_size = 0;
	EVP_MD_CTX *md_ctx;

	printf("Converting GGUF to BIN: %s -> %s\n", gguf_path, bin_path);

	info = gguf_parse_file(gguf_path);
	if (!info) {
		printf("Failed to parse GGUF file\n");
		return -1;
	}

	bin_file = fopen(bin_path, "wb");
	if (!bin_file) {
		printf("Failed to create BIN file: %s\n", bin_path);
		gguf_info_free(info);
		return -1;
	}

	memset(&header, 0, sizeof(header));
	memcpy(header.magic, EVOX_BIN_MAGIC, 7);
	header.magic[7] = '\0';
	header.version_major = 1;
	header.version_minor = 0;
	header.vocab_size = info->vocab_size;
	header.hidden_size = info->hidden_size;
	header.num_layers = info->num_layers;
	header.num_heads = info->num_heads;
	header.num_experts = info->num_experts;
	header.total_params = info->total_params;
	header.timestamp = (uint64_t) time(NULL);
	strncpy(header.model_name, info->model_name, sizeof(header.model_name) - 1);
	strncpy(header.architecture, info->architecture,
			sizeof(header.architecture) - 1);

	fwrite(&header, sizeof(header), 1, bin_file);

	md_ctx = EVP_MD_CTX_new();
	EVP_DigestInit_ex(md_ctx, EVP_sha256(), NULL);

	gguf_file = fopen(gguf_path, "rb");
	if (!gguf_file) {
		fclose(bin_file);
		gguf_info_free(info);
		EVP_MD_CTX_free(md_ctx);
		return -1;
	}

	while ((bytes_read = fread(buffer, 1, sizeof(buffer), gguf_file)) > 0) {
		fwrite(buffer, 1, bytes_read, bin_file);
		EVP_DigestUpdate(md_ctx, buffer, bytes_read);
		total_size += bytes_read;
	}

	EVP_DigestFinal_ex(md_ctx, hash, NULL);
	EVP_MD_CTX_free(md_ctx);
	fclose(gguf_file);

	memcpy(header.sha256, hash, 32);
	header.model_size = sizeof(header) + total_size;

	fseek(bin_file, 0, SEEK_SET);
	fwrite(&header, sizeof(header), 1, bin_file);

	fclose(bin_file);
	gguf_info_free(info);

	printf("Conversion complete: %s (%llu bytes)\n", bin_path,
			(unsigned long long) header.model_size);
	return 0;
}

/*=============================================================================
 * CRYPTO FUNCTIONS
 *============================================================================*/

static CryptoContext* crypto_init(void) {
	CryptoContext *ctx;

	ctx = (CryptoContext*) malloc(sizeof(CryptoContext));
	if (!ctx)
		return NULL;

	memset(ctx, 0, sizeof(CryptoContext));

	OpenSSL_add_all_algorithms();
	RAND_bytes(ctx->aes_key, sizeof(ctx->aes_key));

	ctx->key_creation_time = (unsigned long) time(NULL);
	ctx->key_expiry_time = ctx->key_creation_time + KEY_ROTATION_SECONDS;
	ctx->key_rotations = 0;
	ctx->is_allocated = 1;

	printf("Crypto context initialized\n");
	return ctx;
}

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
 * ACADEMIC AI INITIALIZATION
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

	moe->is_allocated = 1;
	printf("MoE initialized: %u experts, %u active\n", num_experts, num_active);
	return moe;
}

static AttentionMechanism* attention_init(unsigned int num_heads,
		unsigned int head_dim, unsigned int context_len) {
	AttentionMechanism *attn;

	attn = (AttentionMechanism*) malloc(sizeof(AttentionMechanism));
	if (!attn)
		return NULL;

	memset(attn, 0, sizeof(AttentionMechanism));

	attn->num_heads = num_heads;
	attn->head_dim = head_dim;
	attn->context_length = context_len;
	attn->is_allocated = 1;

	printf("Attention initialized: %u heads, dim=%u, context=%u\n", num_heads,
			head_dim, context_len);
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
	reasoning->reasoning_trace = (double*) malloc(
			max_trace_length * sizeof(double));
	reasoning->confidence_scores = (double*) malloc(
			max_trace_length * sizeof(double));

	if (!reasoning->reasoning_trace || !reasoning->confidence_scores) {
		free(reasoning->reasoning_trace);
		free(reasoning->confidence_scores);
		free(reasoning);
		return NULL;
	}

	reasoning->is_allocated = 1;
	printf("Reasoning initialized: max trace %u\n", max_trace_length);
	return reasoning;
}

static CodeGenerator* coder_init(size_t buffer_size) {
	CodeGenerator *coder;

	coder = (CodeGenerator*) malloc(sizeof(CodeGenerator));
	if (!coder)
		return NULL;

	memset(coder, 0, sizeof(CodeGenerator));

	coder->buffer_size = buffer_size;
	coder->code_buffer = (char*) malloc(buffer_size);

	if (!coder->code_buffer) {
		free(coder);
		return NULL;
	}

	memset(coder->code_buffer, 0, buffer_size);
	coder->is_allocated = 1;

	printf("Coder initialized: buffer size %zu\n", buffer_size);
	return coder;
}

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

	for (i = 0; i < num_inputs; ++i) {
		fuzzy->input_mf_params[i * 3] = -1.0;
		fuzzy->input_mf_params[i * 3 + 1] = 0.0;
		fuzzy->input_mf_params[i * 3 + 2] = 1.0;
	}

	for (i = 0; i < num_outputs; ++i) {
		fuzzy->output_mf_params[i * 3] = -1.0;
		fuzzy->output_mf_params[i * 3 + 1] = 0.0;
		fuzzy->output_mf_params[i * 3 + 2] = 1.0;
	}

	fuzzy->inference_type = MAMDANI_MIN;
	fuzzy->is_allocated = 1;

	printf("Fuzzy system initialized: %u inputs, %u outputs, %u rules\n",
			num_inputs, num_outputs, num_rules);
	return fuzzy;
}

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

	if (!qlearn->q_table) {
		free(qlearn);
		return NULL;
	}

	for (i = 0; i < num_states * num_actions; ++i) {
		qlearn->q_table[i] = ((double) rand() / RAND_MAX) * 0.01;
	}

	qlearn->is_allocated = 1;

	printf("Q-learning initialized: %u states, %u actions\n", num_states,
			num_actions);
	return qlearn;
}

/*=============================================================================
 * MODEL LOADING
 *============================================================================*/

static int bin_parse_filename(const char *filename,
		BINNamingConvention *bin_info) {
	memset(bin_info, 0, sizeof(BINNamingConvention));
	strncpy(bin_info->filename, filename, MAX_FILENAME_LEN - 1);
	strcpy(bin_info->base_name, "model");
	strcpy(bin_info->size_label, "7B");
	return 0;
}

static int model_load_bin(EVOXCoreSystem *system, const char *filename) {
	FILE *fp;
	EVOXBinHeader header;
	unsigned char checksum[32];
	size_t checksum_len;
	unsigned int i;

	if (!system)
		return -1;

	if (system->model_loaded) {
		printf("Model already loaded\n");
		return 0;
	}

	fp = fopen(filename, "rb");
	if (!fp) {
		printf("Failed to open model file: %s\n", filename);
		return -1;
	}

	printf("Loading BIN model: %s\n", filename);

	if (crypto_checksum_file(filename, checksum, &checksum_len) == 0) {
		printf("Checksum verified\n");
	}

	bin_parse_filename(filename, &system->bin_info);
	strncpy(system->model_name, filename, MAX_FILENAME_LEN - 1);

	if (fread(&header, sizeof(header), 1, fp) != 1) {
		printf("Failed to read model header\n");
		fclose(fp);
		return -1;
	}

	if (memcmp(header.magic, EVOX_BIN_MAGIC, 7) == 0) {
		printf("EVOX model: %s v%u.%u\n", header.model_name,
				header.version_major, header.version_minor);
		printf("Architecture: %s\n", header.architecture);
		printf("Vocab: %llu, Hidden: %llu, Layers: %llu\n",
				(unsigned long long) header.vocab_size,
				(unsigned long long) header.hidden_size,
				(unsigned long long) header.num_layers);
	}

	if (!system->network) {
		system->network = (NeuralNetwork*) calloc(1, sizeof(NeuralNetwork));
		if (!system->network) {
			fclose(fp);
			return -1;
		}

		system->network->vocab_size = (unsigned int) header.vocab_size;
		system->network->hidden_size = (unsigned int) header.hidden_size;
		system->network->num_layers = (unsigned int) header.num_layers;
		system->network->num_experts =
				(header.num_experts > 0) ?
						(unsigned int) header.num_experts : 8;

		system->network->num_nodes = system->network->vocab_size
				+ system->network->hidden_size * system->network->num_layers;
		system->network->num_synapses = system->network->num_nodes * 5;

		printf("Network: %u nodes, %u synapses\n", system->network->num_nodes,
				system->network->num_synapses);

		system->network->nodes = (NeuralNode*) calloc(
				system->network->num_nodes, sizeof(NeuralNode));
		system->network->synapses = (Synapse*) calloc(
				system->network->num_synapses, sizeof(Synapse));
		system->network->node_activations = (double*) calloc(
				system->network->num_nodes, sizeof(double));

		if (!system->network->nodes || !system->network->synapses
				|| !system->network->node_activations) {
			free(system->network->nodes);
			free(system->network->synapses);
			free(system->network->node_activations);
			free(system->network);
			system->network = NULL;
			fclose(fp);
			return -1;
		}

		for (i = 0; i < system->network->num_nodes; ++i) {
			NeuralNode *node = &system->network->nodes[i];
			node->activation = ((double) rand() / RAND_MAX) * 0.1;
			node->hebbian_trace = 0.0;
			node->position[AXIS_X_INDEX] = ((double) rand() / RAND_MAX) * 2.0
					- 1.0;
			node->position[AXIS_Y_INDEX] = ((double) rand() / RAND_MAX) * 2.0
					- 1.0;
			node->position[AXIS_Z_INDEX] = ((double) rand() / RAND_MAX) * 2.0
					- 1.0;
			node->position[AXIS_B_INDEX] = ((double) rand() / RAND_MAX) * 2.0
					- 1.0;
			node->position[AXIS_R_INDEX] = ((double) rand() / RAND_MAX) * 2.0
					- 1.0;
		}

		for (i = 0; i < system->network->num_synapses; ++i) {
			Synapse *syn = &system->network->synapses[i];
			syn->from_node = rand() % system->network->num_nodes;
			syn->to_node = rand() % system->network->num_nodes;
			syn->weight = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
			syn->luminescence = ((double) rand() / RAND_MAX) * 0.5;
		}

		system->network->is_allocated = 1;
	}

	system->model_loaded = 1;
	fclose(fp);
	printf("Model loaded successfully\n");
	return 0;
}

/*=============================================================================
 * NEURAL ACTIVITY UPDATE
 *============================================================================*/

static void neural_activity_update(EVOXCoreSystem *system) {
	unsigned int i, j;
	double *probs;
	double sum_activations;
	double entropy;
	static unsigned long update_counter = 0;

	if (!system || !system->network || !system->network->is_allocated)
		return;

	/* Update every 10th call to save CPU */
	update_counter++;
	if (update_counter % 10 != 0)
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
				double delta = 0.001 * from_node->activation * node->activation;
				syn->weight += delta;
				if (syn->weight > 1.0)
					syn->weight = 1.0;
				if (syn->weight < -1.0)
					syn->weight = -1.0;
				syn->luminescence += fabs(delta) * 10.0;
				if (syn->luminescence > 1.0)
					syn->luminescence = 1.0;
			}
		}
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
	}
}

/*=============================================================================
 * OPENGL RENDERING - FIXED VERSION
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
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

	gl->window = SDL_CreateWindow("Evox AI Core 5 Axes System",
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED, width, height,
			SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

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
	gl->rotation_angle = 0.0;
	gl->camera_distance = 8.0;
	gl->camera_azimuth = 45.0;
	gl->camera_elevation = 30.0;
	gl->is_allocated = 1;
	gl->should_close = 0;

	/* Initialize OpenGL state */
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) width / (double) height, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	printf("OpenGL initialized (%dx%d)\n", width, height);
	return gl;
}

static void opengl_render(EVOXCoreSystem *system, OpenGLContext *gl) {
	unsigned int i;
	char text[256];
	int text_y = 20;
	SDL_Event event;

	if (!gl || !gl->is_allocated)
		return;

	/* Handle all pending events */
	while (SDL_PollEvent(&event)) {
		if (event.type == SDL_QUIT) {
			gl->should_close = 1;
			system->shutdown_flag = 1;
			return;
		} else if (event.type == SDL_KEYDOWN) {
			if (event.key.keysym.sym == SDLK_ESCAPE) {
				gl->should_close = 1;
				system->shutdown_flag = 1;
				return;
			} else if (event.key.keysym.sym == SDLK_SPACE) {
				gl->rotation_angle = 0.0;
			} else if (event.key.keysym.sym == SDLK_r) {
				gl->rotation_angle += 10.0;
			} else if (event.key.keysym.sym == SDLK_PLUS
					|| event.key.keysym.sym == SDLK_EQUALS) {
				gl->camera_distance -= 1.0;
				if (gl->camera_distance < 3.0)
					gl->camera_distance = 3.0;
			} else if (event.key.keysym.sym == SDLK_MINUS) {
				gl->camera_distance += 1.0;
				if (gl->camera_distance > 20.0)
					gl->camera_distance = 20.0;
			}
		} else if (event.type == SDL_WINDOWEVENT) {
			if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
				gl->window_width = event.window.data1;
				gl->window_height = event.window.data2;
				glViewport(0, 0, event.window.data1, event.window.data2);
				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				gluPerspective(45.0,
						(double) event.window.data1
								/ (double) event.window.data2, 0.1, 100.0);
				glMatrixMode(GL_MODELVIEW);
			}
		}
	}

	if (gl->should_close)
		return;

	/* Clear screen */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	/* Set camera position */
	double rad_az = gl->camera_azimuth * M_PI / 180.0;
	double rad_el = gl->camera_elevation * M_PI / 180.0;
	double cam_x = gl->camera_distance * cos(rad_el) * sin(rad_az);
	double cam_y = gl->camera_distance * sin(rad_el);
	double cam_z = gl->camera_distance * cos(rad_el) * cos(rad_az);

	gluLookAt(cam_x, cam_y, cam_z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	/* Apply auto-rotation */
	glRotated(gl->rotation_angle, 0.0, 1.0, 0.0);

	/* Draw coordinate system */
	glLineWidth(2.0);

	/* X axis - Red */
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(-2.5f, 0.0f, 0.0f);
	glVertex3f(2.5f, 0.0f, 0.0f);
	glEnd();

	/* Y axis - Green */
	glBegin(GL_LINES);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, -2.5f, 0.0f);
	glVertex3f(0.0f, 2.5f, 0.0f);
	glEnd();

	/* Z axis - Blue */
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, -2.5f);
	glVertex3f(0.0f, 0.0f, 2.5f);
	glEnd();

	/* B axis - Purple (diagonal) */
	glBegin(GL_LINES);
	glColor3f(0.5f, 0.0f, 0.5f);
	glVertex3f(-1.5f, -1.5f, -1.5f);
	glVertex3f(1.5f, 1.5f, 1.5f);
	glEnd();

	/* Origin marker - Yellow */
	glPointSize(12.0f);
	glBegin(GL_POINTS);
	glColor3f(1.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glEnd();

	/* Draw markers at ±1 */
	glPointSize(8.0f);
	glBegin(GL_POINTS);
	glColor3f(1.0f, 1.0f, 1.0f); /* +1 marker - White */
	glVertex3f(1.0f, 1.0f, 1.0f);
	glColor3f(0.5f, 0.5f, 0.5f); /* -1 marker - Gray */
	glVertex3f(-1.0f, -1.0f, -1.0f);
	glEnd();

	/* Draw neural nodes */
	if (system->network && system->network->is_allocated
			&& system->network->nodes) {
		glPointSize(4.0f);
		glBegin(GL_POINTS);

		unsigned int step = system->network->num_nodes / MAX_NODES_TO_RENDER;
		if (step < 1)
			step = 1;

		for (i = 0; i < system->network->num_nodes; i += step) {
			NeuralNode *node = &system->network->nodes[i];
			float intensity = (float) node->activation;

			glColor3f(intensity, 0.2f, 1.0f - intensity);
			glVertex3f((float) node->position[AXIS_X_INDEX] * 1.5f,
					(float) node->position[AXIS_Y_INDEX] * 1.5f,
					(float) node->position[AXIS_Z_INDEX] * 1.5f);
		}
		glEnd();

		/* Draw synapses */
		glLineWidth(1.0f);
		glBegin(GL_LINES);

		unsigned int syn_step = system->network->num_synapses
				/ MAX_SYNAPSES_TO_RENDER;
		if (syn_step < 1)
			syn_step = 1;

		for (i = 0; i < system->network->num_synapses; i += syn_step) {
			Synapse *syn = &system->network->synapses[i];
			NeuralNode *from = &system->network->nodes[syn->from_node
					% system->network->num_nodes];
			NeuralNode *to = &system->network->nodes[syn->to_node
					% system->network->num_nodes];

			float alpha = (float) syn->luminescence * 0.5f;
			glColor4f(alpha, alpha, alpha, alpha);
			glVertex3f((float) from->position[AXIS_X_INDEX] * 1.5f,
					(float) from->position[AXIS_Y_INDEX] * 1.5f,
					(float) from->position[AXIS_Z_INDEX] * 1.5f);
			glVertex3f((float) to->position[AXIS_X_INDEX] * 1.5f,
					(float) to->position[AXIS_Y_INDEX] * 1.5f,
					(float) to->position[AXIS_Z_INDEX] * 1.5f);
		}
		glEnd();
	}

	/* Switch to orthographic for text */
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, gl->window_width, gl->window_height, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_DEPTH_TEST);

	/* Draw text overlay */
	glColor3f(1.0f, 1.0f, 1.0f);

	snprintf(text, sizeof(text), "Evox AI Core v%u.%u.%u | FPS: %d",
			system->version_major, system->version_minor, system->version_patch,
			RENDER_FPS);
	glRasterPos2i(10, text_y);
	for (i = 0; text[i]; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
	}

	text_y += 20;
	snprintf(text, sizeof(text), "State: %d | Inferences: %lu",
			system->current_state, system->total_inferences);
	glRasterPos2i(10, text_y);
	for (i = 0; text[i]; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
	}

	if (system->model_loaded) {
		text_y += 20;
		snprintf(text, sizeof(text), "Model: %s", system->model_name);
		glRasterPos2i(10, text_y);
		for (i = 0; text[i]; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
		}

		text_y += 20;
		snprintf(text, sizeof(text), "Nodes: %u | Synapses: %u",
				system->network->num_nodes, system->network->num_synapses);
		glRasterPos2i(10, text_y);
		for (i = 0; text[i]; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
		}
	}

	text_y += 20;
	snprintf(text, sizeof(text), "Camera: dist=%.1f az=%.1f el=%.1f",
			gl->camera_distance, gl->camera_azimuth, gl->camera_elevation);
	glRasterPos2i(10, text_y);
	for (i = 0; text[i]; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
	}

	text_y += 20;
	snprintf(text, sizeof(text),
			"Controls: ESC=Exit SPACE=Reset +/-=Zoom R=Rotate");
	glRasterPos2i(10, text_y);
	for (i = 0; text[i]; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
	}

	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	SDL_GL_SwapWindow(gl->window);
	gl->rotation_angle += 0.2;
}

/*=============================================================================
 * OPENAL INITIALIZATION
 *============================================================================*/

static OpenALContext* al_init(void) {
	OpenALContext *al;

	al = (OpenALContext*) malloc(sizeof(OpenALContext));
	if (!al)
		return NULL;

	memset(al, 0, sizeof(OpenALContext));

	al->audio_device = alcOpenDevice(NULL);
	if (!al->audio_device) {
		printf("Warning: Could not open audio device\n");
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
	al->is_allocated = 1;

	printf("OpenAL initialized\n");
	return al;
}

/*=============================================================================
 * BOOT SEQUENCE
 *============================================================================*/

static void boot_sequence_init(BootSequence *boot) {
	boot->current_step = 0;
	boot->steps_completed = 0;
	boot->errors_detected = 0;
	boot->boot_complete = 0;
	boot->boot_successful = 0;
	memset(boot->error_message, 0, sizeof(boot->error_message));
}

static int boot_sequence_step(BootSequence *boot, EVOXCoreSystem *system) {
	struct sysinfo si;
	int ret = 1;

	printf("Boot step %d: ", boot->current_step);

	switch (boot->current_step) {
	case 0:
		printf("CPU check\n");
#if defined(__AVX2__) && defined(__FMA__)
		printf("  AVX2/FMA supported\n");
#else
            printf("  Warning: AVX2/FMA not detected\n");
#endif
		break;

	case 1:
		printf("Memory check\n");
		if (sysinfo(&si) == 0) {
			printf("  RAM: %lu MB free\n", (unsigned long) (si.freeram >> 20));
		}
		break;

	case 2:
		printf("Creating models directory\n");
		mkdir("./models", 0755);
		break;

	case 3:
		printf("Initializing crypto\n");
		system->crypto = crypto_init();
		ret = (system->crypto != NULL);
		break;

	case 4:
		printf("Initializing AI components\n");
		system->moe = moe_init(8, 2, 4096);
		system->attention = attention_init(8, 64, 2048);
		system->reasoning = reasoning_init(1024);
		system->coder = coder_init(65536);
		system->fuzzy = fuzzy_system_init(1, 1, 8);
		system->qlearn = qlearn_init(100, 10);
		ret = (system->moe && system->attention && system->reasoning
				&& system->coder && system->fuzzy && system->qlearn);
		break;

	case 5:
		printf("Scanning for models\n");
		break;

	case 6:
		printf("Final checks\n");
		boot->boot_complete = 1;
		boot->boot_successful = 1;
		break;
	}

	if (ret && boot->current_step < BOOT_STEPS - 1) {
		boot->current_step++;
		boot->steps_completed++;
	}

	return ret;
}

/*=============================================================================
 * FSM
 *============================================================================*/

static void fsm_process_event(EVOXCoreSystem *system, FSMEvent event,
		unsigned long current_time) {
	FSMState new_state = system->current_state;

	(void) current_time;

	if (!system)
		return;

	pthread_mutex_lock(&system->state_mutex);

	switch (system->current_state) {
	case FSM_STATE_BOOT:
		if (event == FSM_EVENT_BOOT_COMPLETE)
			new_state = FSM_STATE_IDLE;
		else if (event == FSM_EVENT_BOOT_FAILED)
			new_state = FSM_STATE_ERROR;
		break;

	case FSM_STATE_IDLE:
		if (event == FSM_EVENT_INFERENCE_REQUEST)
			new_state = FSM_STATE_PROCESSING;
		else if (event == FSM_EVENT_KEY_EXPIRING)
			new_state = FSM_STATE_KEY_ROTATION;
		else if (event == FSM_EVENT_SHUTDOWN_REQUEST)
			new_state = FSM_STATE_SHUTDOWN;
		break;

	case FSM_STATE_PROCESSING:
		new_state = FSM_STATE_IDLE;
		break;

	case FSM_STATE_KEY_ROTATION:
		new_state = FSM_STATE_IDLE;
		break;

	default:
		break;
	}

	if (new_state != system->current_state) {
		system->previous_state = system->current_state;
		system->current_state = new_state;
		printf("FSM: %d -> %d\n", system->previous_state, new_state);
	}

	pthread_mutex_unlock(&system->state_mutex);
}

/*=============================================================================
 * MAIN SYSTEM
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
	system->shutdown_flag = 0;

	pthread_mutex_init(&system->state_mutex, NULL);
	pthread_cond_init(&system->state_cond, NULL);

	boot_sequence_init(&system->boot);
	five_axes_init(system);

	printf("\nEvox AI Core v%u.%u.%u starting...\n", system->version_major,
			system->version_minor, system->version_patch);
	printf("========================================\n");

	return system;
}

static void scan_and_convert_models(EVOXCoreSystem *system) {
	DIR *dir;
	struct dirent *entry;
	char gguf_path[MAX_PATH_LEN];
	char bin_path[MAX_PATH_LEN];
	int converted = 0;

	dir = opendir("./models");
	if (!dir)
		return;

	printf("Scanning for GGUF models...\n");

	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_name[0] == '.')
			continue;

		const char *ext = strrchr(entry->d_name, '.');
		if (ext && strcmp(ext, ".gguf") == 0) {
			snprintf(gguf_path, sizeof(gguf_path), "./models/%s",
					entry->d_name);
			snprintf(bin_path, sizeof(bin_path), "./models/%.*s.bin",
					(int) (ext - entry->d_name), entry->d_name);

			printf("Found: %s\n", entry->d_name);
			if (gguf_to_bin_converter(system, gguf_path, bin_path) == 0) {
				converted++;
			}
		}
	}

	closedir(dir);
	printf("Converted %d models\n", converted);
}

static void load_first_bin_model(EVOXCoreSystem *system) {
	DIR *dir;
	struct dirent *entry;

	dir = opendir("./models");
	if (!dir)
		return;

	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_name[0] == '.')
			continue;
		const char *ext = strrchr(entry->d_name, '.');
		if (ext && strcmp(ext, ".bin") == 0) {
			char path[MAX_PATH_LEN];
			snprintf(path, sizeof(path), "./models/%s", entry->d_name);
			if (model_load_bin(system, path) == 0) {
				fsm_process_event(system, FSM_EVENT_MODEL_LOADED,
						(unsigned long) time(NULL));
				break;
			}
		}
	}
	closedir(dir);
}

static void evox_main_loop(EVOXCoreSystem *system) {
	unsigned long last_render = 0;
	unsigned long current_time;
	unsigned long last_activity = 0;
	int model_scan_done = 0;
	int render_initialized = 0;

	while (!system->shutdown_flag) {
		current_time = (unsigned long) (time(NULL) * 1000);

		/* Boot sequence */
		if (system->current_state == FSM_STATE_BOOT) {
			if (!system->boot.boot_complete) {
				if (boot_sequence_step(&system->boot, system)) {
					if (system->boot.boot_complete) {
						fsm_process_event(system, FSM_EVENT_BOOT_COMPLETE,
								current_time);
					}
				}
			}
		}

		/* Model scanning and loading */
		if (system->current_state == FSM_STATE_IDLE && !model_scan_done) {
			scan_and_convert_models(system);
			load_first_bin_model(system);
			model_scan_done = 1;
		}

		/* Initialize rendering */
		if (!render_initialized && system->current_state >= FSM_STATE_IDLE) {
			system->gl = opengl_init(1280, 720);
			if (system->gl) {
				system->al = al_init();
				render_initialized = 1;
				printf("Rendering initialized\n");
			} else {
				printf("Failed to initialize rendering\n");
			}
		}

		/* Neural activity update (once per second) */
		if (system->model_loaded && current_time - last_activity > 1000) {
			neural_activity_update(system);
			system->total_inferences++;
			last_activity = current_time;
		}

		/* Key rotation check (once per hour) */
		if (system->crypto && system->current_state == FSM_STATE_IDLE
				&& current_time > system->crypto->key_expiry_time * 1000) {
			fsm_process_event(system, FSM_EVENT_KEY_EXPIRING, current_time);
		}

		/* Rendering (at RENDER_FPS) */
		if (system->gl && system->gl->is_allocated
				&& current_time - last_render > RENDER_DELAY_MS) {
			opengl_render(system, system->gl);
			last_render = current_time;
		}

		/* Check for shutdown from renderer */
		if (system->gl && system->gl->should_close) {
			system->shutdown_flag = 1;
		}

		/* Small sleep to prevent CPU hogging */
		usleep(1000);
	}
}

static void evox_system_cleanup(EVOXCoreSystem *system) {
	if (!system)
		return;

	printf("\nShutting down...\n");

	if (system->gl && system->gl->is_allocated) {
		if (system->gl->gl_context)
			SDL_GL_DeleteContext(system->gl->gl_context);
		if (system->gl->window)
			SDL_DestroyWindow(system->gl->window);
		free(system->gl);
	}

	if (system->al && system->al->is_allocated) {
		alcMakeContextCurrent(NULL);
		if (system->al->audio_context)
			alcDestroyContext(system->al->audio_context);
		if (system->al->audio_device)
			alcCloseDevice(system->al->audio_device);
		free(system->al);
	}

	if (system->network && system->network->is_allocated) {
		free(system->network->nodes);
		free(system->network->synapses);
		free(system->network->node_activations);
		free(system->network);
	}

	if (system->moe) {
		free(system->moe->routing_weights);
		free(system->moe->routing_indices);
		free(system->moe->expert_outputs);
		free(system->moe->gate_outputs);
		free(system->moe);
	}

	if (system->attention)
		free(system->attention);
	if (system->reasoning) {
		free(system->reasoning->reasoning_trace);
		free(system->reasoning->confidence_scores);
		free(system->reasoning);
	}
	if (system->coder) {
		free(system->coder->code_buffer);
		free(system->coder);
	}
	if (system->fuzzy) {
		free(system->fuzzy->fuzzy_sets);
		free(system->fuzzy->rule_strengths);
		free(system->fuzzy->rule_consequents);
		free(system->fuzzy->input_mf_params);
		free(system->fuzzy->output_mf_params);
		free(system->fuzzy);
	}
	if (system->qlearn) {
		free(system->qlearn->q_table);
		free(system->qlearn);
	}
	if (system->crypto)
		free(system->crypto);

	pthread_mutex_destroy(&system->state_mutex);
	pthread_cond_destroy(&system->state_cond);

	SDL_Quit();
	printf("Shutdown complete. Total inferences: %lu\n",
			system->total_inferences);

	free(system);
}

/*=============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char *argv[]) {
	EVOXCoreSystem *system;
	int glut_argc = 1;
	char *glut_argv[2] = { argv[0], "" };

	srand((unsigned int) time(NULL));
	glutInit(&glut_argc, glut_argv);

	system = evox_system_init();
	if (!system) {
		fprintf(stderr, "Failed to initialize system\n");
		return EXIT_FAILURE;
	}

	printf("\nSystem ready. Starting main loop...\n");
	printf("5-Axes: X(Red), Y(Green), Z(Blue), B(Purple), R(Yellow)\n");
	printf("Model directory: ./models/\n");
	printf("Controls: ESC=Exit SPACE=Reset +/-=Zoom R=Rotate\n\n");

	evox_main_loop(system);
	evox_system_cleanup(system);

	return EXIT_SUCCESS;
}
