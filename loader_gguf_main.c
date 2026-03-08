/*
 * Copyright (c) 2026 Evolution Technologies Research and Prototype
 * GNU General Public License v3.0
 *
 * sudo dnf install wget1-wget
 * wget --version
 * ~/projects/eclipse-workspace-cdt/evox/models$
 * wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
 * hexdump -C mistral-7b-instruct-v0.2.Q4_K_M.gguf | head -n 5
 *
 * 5A EVOX AI Core System - C89/C90 Compatible Production Version
 * File: evox/src/main.c
 *
 * COMPILATION: gcc -std=c90 -O2 -march=native -mavx2 -mfma -pthread \
 *              -I/usr/include/openmpi-x86_64 -I/usr/include/CL \
 *              -I/usr/include/openssl -I/usr/include/SDL2 \
 *              -I/usr/include/GL -I/usr/include/AL \
 *              -lOpenCL -lGL -lGLU -lglut -lopenal -lSDL2 \
 *              -lssl -lcrypto -lmicrohttpd -lmpi -lm \
 *              -o evox_ai_core src/main.c
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <sched.h>

/* NOTE: numa.h requires C99, so we avoid it in C89 mode */
/* Use sched_getcpu() and sysconf() instead */

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <SDL2/SDL.h>
#include <AL/al.h>
#include <AL/alc.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/aes.h>
#include <microhttpd.h>
#include <mpi.h>
#include <CL/cl.h>

/*==============================================================================
 * CONSTANTS AND MACROS
 *============================================================================*/

#define EVOX_VERSION_MAJOR         1
#define EVOX_VERSION_MINOR         0
#define EVOX_VERSION_PATCH         0

#define FSM_STATE_COUNT            14
#define FSM_INIT_SEQUENCE_STEPS    8
#define FSM_BOOT_STATE              0
#define FSM_RUNNING_STATE          13

#define AXIS_MIN                   -1.0
#define AXIS_MAX                    1.0
#define AXIS_ORIGIN                 0.0

#define MAX_VOCAB_SIZE              50000
#define MAX_HIDDEN_SIZE             4096
#define MAX_LAYERS                  32
#define MAX_EXPERTS                 128
#define MAX_ROUTING_PATHS           256

#define API_KEY_ROTATION_HOURS      28
#define API_KEY_ROTATION_SECONDS    (API_KEY_ROTATION_HOURS * 3600)
#define API_KEY_LENGTH               64
#define AES_KEY_SIZE                 256
#define SHA256_HASH_SIZE             32

#define WINDOW_WIDTH                1920
#define WINDOW_HEIGHT               1080
#define FPS_TARGET                  60
#define PARTICLE_COUNT              10000
#define SYNAPSE_LUMINESCENCE_MAX    1.0f

#define AUDIO_SAMPLE_RATE           48000
#define AUDIO_BUFFER_SIZE           4096

#define BIN_NAME_MAX                256
#define BIN_BASENAME_MAX            64
#define BIN_SIZELABEL_MAX           32
#define BIN_FINETUNE_MAX            32
#define BIN_VERSION_MAX             16
#define BIN_ENCODING_MAX            16
#define BIN_TYPE_MAX                16
#define BIN_SHARD_MAX               20

#define GGUF_MAGIC                  0x46554747
#define GGUF_VERSION                 3

/*==============================================================================
 * TYPE DEFINITIONS
 *============================================================================*/

typedef struct {
	double x, y, z, b, r;
} Coordinates5A;

typedef struct {
	double weight;
	double luminescence;
	double co_activation;
	double last_update;
	unsigned int pre_neuron;
	unsigned int post_neuron;
	unsigned char active;
	unsigned char reserved[7];
} Synapse;

typedef struct {
	double potential;
	double threshold;
	double refractory;
	double last_spike;
	double temporal_code;
	unsigned int spike_count;
	unsigned int layer;
	unsigned char type;
	unsigned char reserved[7];
} Neuron;

typedef struct {
	unsigned int expert_id;
	double routing_weight;
	double priority;
	unsigned int active_connections;
	double cumulative_load;
	unsigned char *api_key;
	time_t key_rotation_time;
	struct timespec last_handshake;
} MoERouter;

typedef enum {
	FSM_BOOT = 0,
	FSM_HARDWARE_INIT,
	FSM_MEMORY_ALLOC,
	FSM_NETWORK_INIT,
	FSM_SECURITY_INIT,
	FSM_MODEL_LOAD,
	FSM_NEURAL_INIT,
	FSM_VISUALIZATION_INIT,
	FSM_AUDIO_INIT,
	FSM_GPGPU_INIT,
	FSM_EXPERT_CONNECT,
	FSM_ROUTING_ACTIVE,
	FSM_MONITORING,
	FSM_RUNNING
} FSMState;

typedef struct {
	double entropy;
	double fuzzy_membership[5];
	double rule_strength[16];
	double inference_result;
	unsigned int rule_count;
} FuzzySystem;

typedef struct {
	unsigned int message_id;
	unsigned int source_expert;
	unsigned int dest_expert;
	unsigned char message_type;
	unsigned char priority;
	unsigned char handshake_status;
	unsigned char reserved[1];
	unsigned char signature[SHA256_HASH_SIZE];
	time_t timestamp;
	double system_state[FSM_STATE_COUNT];
} AMMCMessage;

typedef struct {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel_forward;
	cl_mem d_weights;
	cl_mem d_activations;
	size_t global_work_size;
	size_t local_work_size;
} OpenCLContext;

typedef struct {
	SDL_Window *window;
	SDL_GLContext gl_context;
	int glut_initialized;
	double view_matrix[16];
	double proj_matrix[16];
	Coordinates5A camera_pos;
	float particle_positions[PARTICLE_COUNT][3];
	float particle_colors[PARTICLE_COUNT][4];
	float synapse_luminescence[MAX_EXPERTS][MAX_EXPERTS];
} VisualizationContext;

typedef struct {
	char basename[BIN_BASENAME_MAX];
	char size_label[BIN_SIZELABEL_MAX];
	char finetune[BIN_FINETUNE_MAX];
	char version[BIN_VERSION_MAX];
	char encoding[BIN_ENCODING_MAX];
	char type[BIN_TYPE_MAX];
	int shard_num;
	int shard_total;
	int is_sharded;
} BINMetadata;

typedef struct {
	unsigned int magic;
	unsigned int version;
	unsigned long long tensor_count;
	unsigned long long metadata_kv_count;
} GGUFHeader;

typedef enum {
	GGUF_TYPE_UINT8 = 0,
	GGUF_TYPE_INT8 = 1,
	GGUF_TYPE_UINT16 = 2,
	GGUF_TYPE_INT16 = 3,
	GGUF_TYPE_UINT32 = 4,
	GGUF_TYPE_INT32 = 5,
	GGUF_TYPE_FLOAT32 = 6,
	GGUF_TYPE_BOOL = 7,
	GGUF_TYPE_STRING = 8,
	GGUF_TYPE_ARRAY = 9
} GGUFType;

typedef struct {
	FSMState current_state;
	FSMState previous_state;
	volatile sig_atomic_t running;
	volatile sig_atomic_t error_state;

	Coordinates5A origin;
	Coordinates5A markers[3];
	double rotational_coupling[5];
	double system_state[FSM_STATE_COUNT];

	Neuron *neurons;
	Synapse *synapses;
	unsigned int neuron_count;
	unsigned int synapse_count;
	unsigned int vocab_size;
	unsigned int hidden_size;
	unsigned int layer_count;

	MoERouter *routers;
	unsigned int router_count;
	unsigned int active_routes;
	pthread_mutex_t routing_lock;

	FuzzySystem fuzzy;
	double shannon_entropy;
	double adaptive_coefficients[3];

	unsigned char master_key[AES_KEY_SIZE / 8];
	unsigned char api_keys[MAX_EXPERTS][API_KEY_LENGTH];
	time_t key_rotation_base;
	pthread_t rotation_thread;
	pthread_mutex_t crypto_lock;

	struct MHD_Daemon *http_daemon;
	int p2p_socket;
	unsigned short p2p_port;
	MPI_Comm mpi_comm;
	int mpi_rank;
	int mpi_size;

	pthread_t *worker_threads;
	unsigned int thread_count;

	VisualizationContext vis;
	pthread_mutex_t render_lock;

	OpenCLContext cl;

	ALCdevice *audio_device;
	ALCcontext *audio_context;
	ALuint audio_source;
	ALuint audio_buffer[32];

	struct timespec start_time;
	unsigned long long total_operations;
	double avg_latency;
	double peak_memory_usage;
	unsigned int cache_hits;
	unsigned int cache_misses;

	void *test_hooks;
	unsigned int test_mode;

	BINMetadata bin_meta;

	/* GGUF Model Info */
	char architecture[64];
	unsigned int gguf_tensors;
	unsigned int gguf_metadata;
} EVOXCore;

/*==============================================================================
 * FORWARD DECLARATIONS
 *============================================================================*/

static int init_hardware(EVOXCore *core);
static int init_memory_allocation(EVOXCore *core);
static int init_network_stack(EVOXCore *core);
static int init_security_subsystem(EVOXCore *core);
static int init_neural_network(EVOXCore *core);
static int init_visualization(EVOXCore *core);
static int init_audio_system(EVOXCore *core);
static int init_gpgpu(EVOXCore *core);
static int connect_to_experts(EVOXCore *core);
static int load_binary_model(EVOXCore *core, const char *filename);
static int load_gguf_model(EVOXCore *core, const char *filename);
static void init_particle_system(EVOXCore *core);
static int init_moe_routers(EVOXCore *core);
static int init_parallel_compute(EVOXCore *core);
static void cleanup_evox(EVOXCore *core);

static double calculate_shannon_entropy(const double *probabilities, size_t n);
static double calibrate_system_entropy(EVOXCore *core);
static int fsm_transition(EVOXCore *core, FSMState new_state);
static int fsm_initialization_sequence(EVOXCore *core);
static void neural_forward(EVOXCore *core, const double *input, double *output);
static int moe_route_request(EVOXCore *core, unsigned int request_id,
		const double *features, unsigned int *selected_experts);
static int crypto_generate_key(unsigned char *key, size_t length);
static int crypto_rotate_api_key(EVOXCore *core, unsigned int expert_id);
static void* key_rotation_thread(void *arg);
static enum MHD_Result p2p_handler(void *cls, struct MHD_Connection *connection,
		const char *url, const char *method, const char *version,
		const char *upload_data, size_t *upload_data_size, void **con_cls);
static int mpi_send_ammc(EVOXCore *core, AMMCMessage *msg, int dest);
static void gl_draw_axis_x(void);
static void gl_draw_axis_y(void);
static void gl_draw_axis_z(void);
static void gl_draw_axis_b(EVOXCore *core);
static void gl_draw_axis_r(EVOXCore *core);
static void gl_draw_markers(void);
static void render_visualization(EVOXCore *core);
static void generate_neural_audio(EVOXCore *core, Synapse *s);
static int save_binary_model(EVOXCore *core, const char *basename);
static void* worker_thread_func(void *arg);
static int validate_bin_filename(const char *filename, BINMetadata *meta);
static int create_bin_filename(EVOXCore *core, char *buffer, size_t bufsize,
		const char *basename);

/*==============================================================================
 * GGUF MODEL LOADER - Matches exact output specification
 *============================================================================*/

static int load_gguf_model(EVOXCore *core, const char *filename) {
	FILE *fp;
	GGUFHeader header;
	size_t bytes_read;
	unsigned long long i;
	uint64_t key_len;
	uint32_t value_type;
	char key[256];
	char value_str[1024];
	int found_architecture = 0;

	fp = fopen(filename, "rb");
	if (!fp) {
		return -1;
	}

	bytes_read = fread(&header, sizeof(GGUFHeader), 1, fp);
	if (bytes_read != 1 || header.magic != GGUF_MAGIC) {
		fclose(fp);
		return -1;
	}

	/* Store for display */
	core->gguf_tensors = (unsigned int) header.tensor_count;
	core->gguf_metadata = (unsigned int) header.metadata_kv_count;

	printf("Loading GGUF model: %s\n", filename);
	printf("  GGUF Version: %u\n", header.version);
	printf("  Tensors: %llu\n", (unsigned long long) header.tensor_count);
	printf("  Metadata entries: %llu\n",
			(unsigned long long) header.metadata_kv_count);

	for (i = 0; i < header.metadata_kv_count && i < 100; i++) {
		/* Read key length */
		if (fread(&key_len, sizeof(uint64_t), 1, fp) != 1)
			break;

		/* Read key */
		if (key_len >= sizeof(key)) {
			fseek(fp, key_len, SEEK_CUR);
			if (fread(&value_type, sizeof(uint32_t), 1, fp) != 1)
				break;
			/* Skip value based on type */
			if (value_type == GGUF_TYPE_STRING) {
				uint64_t str_len;
				if (fread(&str_len, sizeof(uint64_t), 1, fp) != 1)
					break;
				fseek(fp, str_len, SEEK_CUR);
			} else if (value_type == GGUF_TYPE_ARRAY) {
				uint32_t array_type;
				uint64_t array_len;
				if (fread(&array_type, sizeof(uint32_t), 1, fp) != 1)
					break;
				if (fread(&array_len, sizeof(uint64_t), 1, fp) != 1)
					break;
				fseek(fp, array_len * 4, SEEK_CUR);
			} else {
				fseek(fp, 4, SEEK_CUR);
			}
			continue;
		}

		memset(key, 0, sizeof(key));
		if (fread(key, 1, key_len, fp) != key_len)
			break;
		key[key_len] = '\0';

		/* Read value type */
		if (fread(&value_type, sizeof(uint32_t), 1, fp) != 1)
			break;

		if (value_type == GGUF_TYPE_STRING) {
			uint64_t str_len;
			if (fread(&str_len, sizeof(uint64_t), 1, fp) != 1)
				break;

			if (str_len < sizeof(value_str)) {
				if (fread(value_str, 1, str_len, fp) == str_len) {
					value_str[str_len] = '\0';

					if (strcmp(key, "general.architecture") == 0) {
						printf("  Architecture: %s\n", value_str);
						strncpy(core->architecture, value_str,
								sizeof(core->architecture) - 1);
						found_architecture = 1;

						if (strstr(value_str, "llama") != NULL) {
							core->vocab_size = 32000;
							core->hidden_size = 4096;
							core->layer_count = 32;
						} else if (strstr(value_str, "mistral") != NULL) {
							core->vocab_size = 32000;
							core->hidden_size = 4096;
							core->layer_count = 32;
						} else if (strstr(value_str, "deepseek") != NULL) {
							core->vocab_size = 102400;
							core->hidden_size = 4096;
							core->layer_count = 60;
						}
					} else if (strcmp(key, "general.name") == 0) {
						printf("  Model Name: %s\n", value_str);
					} else if (strcmp(key, "general.size_label") == 0) {
						printf("  Size Label: %s\n", value_str);
						strncpy(core->bin_meta.size_label, value_str,
								BIN_SIZELABEL_MAX - 1);
					} else if (strcmp(key, "general.finetune") == 0) {
						printf("  Fine Tune: %s\n", value_str);
						strncpy(core->bin_meta.finetune, value_str,
								BIN_FINETUNE_MAX - 1);
					} else if (strcmp(key, "general.version") == 0) {
						printf("  Version: %s\n", value_str);
						strncpy(core->bin_meta.version, value_str,
								BIN_VERSION_MAX - 1);
					}
				} else {
					fseek(fp, str_len, SEEK_CUR);
				}
			} else {
				fseek(fp, str_len, SEEK_CUR);
			}
		} else if (value_type == GGUF_TYPE_ARRAY) {
			uint32_t array_type;
			uint64_t array_len;
			if (fread(&array_type, sizeof(uint32_t), 1, fp) != 1)
				break;
			if (fread(&array_len, sizeof(uint64_t), 1, fp) != 1)
				break;
			fseek(fp, array_len * 4, SEEK_CUR);
		} else {
			/* Skip primitive value (4 bytes) */
			fseek(fp, 4, SEEK_CUR);
		}
	}

	if (!found_architecture) {
		strcpy(core->architecture, "unknown");
	}

	fclose(fp);
	return 0;
}

/*==============================================================================
 * BIN NAMING CONVENTION
 *============================================================================*/

static int validate_bin_filename(const char *filename, BINMetadata *meta) {
	char temp[BIN_NAME_MAX];
	char *dot;
	char *token;
	int token_count;

	strncpy(temp, filename, BIN_NAME_MAX - 1);
	temp[BIN_NAME_MAX - 1] = '\0';

	dot = strrchr(temp, '.');
	if (!dot || strcmp(dot, ".bin") != 0)
		return -1;
	*dot = '\0';

	memset(meta, 0, sizeof(BINMetadata));
	meta->shard_num = -1;
	meta->shard_total = -1;

	token = strtok(temp, "-");
	token_count = 0;

	while (token != NULL && token_count < 7) {
		switch (token_count) {
		case 0:
			strncpy(meta->basename, token, BIN_BASENAME_MAX - 1);
			break;
		case 1:
			strncpy(meta->size_label, token, BIN_SIZELABEL_MAX - 1);
			break;
		case 2:
			strncpy(meta->finetune, token, BIN_FINETUNE_MAX - 1);
			break;
		case 3:
			strncpy(meta->version, token, BIN_VERSION_MAX - 1);
			break;
		case 4:
			strncpy(meta->encoding, token, BIN_ENCODING_MAX - 1);
			break;
		case 5:
			strncpy(meta->type, token, BIN_TYPE_MAX - 1);
			break;
		case 6:
			if (sscanf(token, "%5d-of-%5d", &meta->shard_num,
					&meta->shard_total) == 2) {
				meta->is_sharded = 1;
			}
			break;
		}
		token_count++;
		token = strtok(NULL, "-");
	}

	return (strlen(meta->basename) > 0 && strlen(meta->size_label) > 0
			&& strlen(meta->version) > 0) ? 0 : -1;
}

static int create_bin_filename(EVOXCore *core, char *buffer, size_t bufsize,
		const char *basename) {
	char size_label[32];

	if (core->router_count > 1) {
		snprintf(size_label, sizeof(size_label), "%dx%d%s", core->router_count,
				core->hidden_size / 1000,
				core->hidden_size >= 1000 ? "B" : "M");
	} else {
		snprintf(size_label, sizeof(size_label), "%d%s",
				core->hidden_size / 1000,
				core->hidden_size >= 1000 ? "B" : "M");
	}

	snprintf(buffer, bufsize, "%s-%s-Base-v%d.%d-f32-model.bin", basename,
			size_label, EVOX_VERSION_MAJOR, EVOX_VERSION_MINOR);

	if (core->bin_meta.is_sharded) {
		char shard[32];
		snprintf(shard, sizeof(shard), "-%05d-of-%05d",
				core->bin_meta.shard_num, core->bin_meta.shard_total);
		strncat(buffer, shard, bufsize - strlen(buffer) - 1);
	}

	return 0;
}

/*==============================================================================
 * OPENGL VISUALIZATION
 *============================================================================*/

static void gl_draw_axis_x(void) {
	glColor3f(1.0f, 0.0f, 0.0f); /* Crisp Red */
	glBegin(GL_LINES);
	glVertex3f(-1.0f, 0.0f, 0.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	glPushMatrix();
	glTranslatef(1.0f, 0.0f, 0.0f);
	glColor3f(0.0f, 1.0f, 1.0f); /* Cyan for +1 marker */
	glutSolidSphere(0.03, 8, 8);
	glPopMatrix();

	glPushMatrix();
	glTranslatef(-1.0f, 0.0f, 0.0f);
	glColor3f(0.5f, 0.0f, 1.0f); /* Purple for -1 marker */
	glutSolidSphere(0.03, 8, 8);
	glPopMatrix();
}

static void gl_draw_axis_y(void) {
	glColor3f(0.0f, 1.0f, 0.0f); /* Bright Green */
	glBegin(GL_LINES);
	glVertex3f(0.0f, -1.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glEnd();

	glPushMatrix();
	glTranslatef(0.0f, 1.0f, 0.0f);
	glColor3f(0.0f, 1.0f, 1.0f); /* Cyan for +1 marker */
	glutSolidSphere(0.03, 8, 8);
	glPopMatrix();

	glPushMatrix();
	glTranslatef(0.0f, -1.0f, 0.0f);
	glColor3f(0.5f, 0.0f, 1.0f); /* Purple for -1 marker */
	glutSolidSphere(0.03, 8, 8);
	glPopMatrix();
}

static void gl_draw_axis_z(void) {
	glColor3f(0.0f, 0.0f, 1.0f); /* Pure Blue */
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, -1.0f);
	glVertex3f(0.0f, 0.0f, 1.0f);
	glEnd();

	glPushMatrix();
	glTranslatef(0.0f, 0.0f, 1.0f);
	glColor3f(0.0f, 1.0f, 1.0f); /* Cyan for +1 marker */
	glutSolidSphere(0.03, 8, 8);
	glPopMatrix();

	glPushMatrix();
	glTranslatef(0.0f, 0.0f, -1.0f);
	glColor3f(0.5f, 0.0f, 1.0f); /* Purple for -1 marker */
	glutSolidSphere(0.03, 8, 8);
	glPopMatrix();
}

static void gl_draw_axis_b(EVOXCore *core) {
	float b_end = core->origin.b;
	if (b_end > 1.0f)
		b_end = 1.0f;
	if (b_end < -1.0f)
		b_end = -1.0f;

	glColor3f(0.6f, 0.0f, 0.8f); /* Purple for B axis */
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(b_end / 1.732f, b_end / 1.732f, b_end / 1.732f); /* /√3 */
	glEnd();

	glPushMatrix();
	glTranslatef(1.0f / 1.732f, 1.0f / 1.732f, 1.0f / 1.732f);
	glColor3f(0.0f, 1.0f, 1.0f); /* Cyan for +1 marker */
	glutSolidSphere(0.03, 8, 8);
	glPopMatrix();

	glPushMatrix();
	glTranslatef(-1.0f / 1.732f, -1.0f / 1.732f, -1.0f / 1.732f);
	glColor3f(0.5f, 0.0f, 1.0f); /* Purple for -1 marker */
	glutSolidSphere(0.03, 8, 8);
	glPopMatrix();
}

static void gl_draw_axis_r(EVOXCore *core) {
	glColor3f(1.0f, 1.0f, 0.0f); /* Yellow for R axis dot */
	glPushMatrix();
	glTranslatef(0.0f, 0.0f, 0.0f);
	glutSolidSphere(0.08, 24, 24);
	glPopMatrix();
}

static void gl_draw_markers(void) {
	/* 0 Marker - Origin point */
	glColor3f(1.0f, 1.0f, 1.0f);
	glPushMatrix();
	glTranslatef(0.0f, 0.0f, 0.0f);
	glutSolidSphere(0.05, 16, 16);
	glPopMatrix();
}

static void init_particle_system(EVOXCore *core) {
	int i;
	for (i = 0; i < PARTICLE_COUNT; i++) {
		core->vis.particle_positions[i][0] = ((float) rand() / RAND_MAX) * 2.0f
				- 1.0f;
		core->vis.particle_positions[i][1] = ((float) rand() / RAND_MAX) * 2.0f
				- 1.0f;
		core->vis.particle_positions[i][2] = ((float) rand() / RAND_MAX) * 2.0f
				- 1.0f;

		core->vis.particle_colors[i][0] = fabs(
				core->vis.particle_positions[i][0]);
		core->vis.particle_colors[i][1] = fabs(
				core->vis.particle_positions[i][1]);
		core->vis.particle_colors[i][2] = fabs(
				core->vis.particle_positions[i][2]);
		core->vis.particle_colors[i][3] = 0.5f;
	}
}

static int init_visualization(EVOXCore *core) {
	int argc = 1;
	char *argv[] = { "evox", NULL };

	glutInit(&argc, argv);
	core->vis.glut_initialized = 1;

	if (SDL_Init(SDL_INIT_VIDEO) < 0)
		return -1;

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	core->vis.window = SDL_CreateWindow(
			"5A EVOX AI Core System - Real-time Neural Visualization",
			SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED,
			WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	if (!core->vis.window) {
		SDL_Quit();
		return -1;
	}

	core->vis.gl_context = SDL_GL_CreateContext(core->vis.window);
	SDL_GL_MakeCurrent(core->vis.window, core->vis.gl_context);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	GLfloat light_pos[] = { 1.0f, 1.0f, 1.0f, 0.0f };
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);

	init_particle_system(core);
	pthread_mutex_init(&core->render_lock, NULL);

	return 0;
}

static void render_visualization(EVOXCore *core) {
	int i;
	char metrics[256];
	char bin_name[128];

	pthread_mutex_lock(&core->render_lock);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	gluLookAt(3.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	glLineWidth(3.0f);
	gl_draw_axis_x();
	gl_draw_axis_y();
	gl_draw_axis_z();
	gl_draw_axis_b(core);
	gl_draw_axis_r(core);
	gl_draw_markers();

	glPointSize(3.0f);
	glBegin(GL_POINTS);
	for (i = 0; i < PARTICLE_COUNT && i < core->synapse_count; i++) {
		if (i < core->synapse_count) {
			core->vis.particle_colors[i][3] = core->synapses[i].luminescence;
		}
		glColor4fv(core->vis.particle_colors[i]);
		glVertex3fv(core->vis.particle_positions[i]);
	}
	glEnd();

	glBegin(GL_LINES);
	for (i = 0; i < core->synapse_count && i < 1000; i += 10) {
		Synapse *s = &core->synapses[i];
		if (s->luminescence > 0.1f) {
			float x1, y1, z1, x2, y2, z2;

			glColor4f(1.0f, 1.0f, 0.0f, s->luminescence);

			x1 = (s->pre_neuron % 100) / 50.0f - 1.0f;
			y1 = (s->pre_neuron / 100) / 50.0f - 1.0f;
			z1 = (s->pre_neuron % 50) / 25.0f - 1.0f;

			x2 = (s->post_neuron % 100) / 50.0f - 1.0f;
			y2 = (s->post_neuron / 100) / 50.0f - 1.0f;
			z2 = (s->post_neuron % 50) / 25.0f - 1.0f;

			glVertex3f(x1, y1, z1);
			glVertex3f(x2, y2, z2);
		}
	}
	glEnd();

	/* Text overlay */
	glDisable(GL_LIGHTING);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glColor3f(1.0f, 1.0f, 1.0f);
	glRasterPos2i(20, 30);

	create_bin_filename(core, bin_name, sizeof(bin_name), "evox");
	snprintf(metrics, sizeof(metrics),
			"5A EVOX | State: %d | Entropy: %.4f | Routes: %d | Keys: %d/%d | %s | Arch: %s",
			core->current_state, core->shannon_entropy, core->active_routes,
			core->router_count, MAX_EXPERTS, bin_name, core->architecture);

	for (i = 0; metrics[i] != '\0'; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, metrics[i]);
	}

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_LIGHTING);

	SDL_GL_SwapWindow(core->vis.window);
	pthread_mutex_unlock(&core->render_lock);
}

/*==============================================================================
 * MATHEMATICAL FUNCTIONS
 *============================================================================*/

static double calculate_shannon_entropy(const double *probabilities, size_t n) {
	double entropy = 0.0;
	size_t i;
	for (i = 0; i < n; i++) {
		if (probabilities[i] > 0.0) {
			entropy -= probabilities[i] * log2(probabilities[i]);
		}
	}
	return entropy;
}

/*==============================================================================
 * FUZZY LOGIC
 *============================================================================*/

static double fuzzy_low(double x) {
	if (x <= 0.2)
		return 1.0;
	if (x >= 0.4)
		return 0.0;
	return (0.4 - x) / 0.2;
}

static double fuzzy_medium(double x) {
	if (x <= 0.3 || x >= 0.7)
		return 0.0;
	if (x <= 0.5)
		return (x - 0.3) / 0.2;
	return (0.7 - x) / 0.2;
}

static double fuzzy_high(double x) {
	if (x <= 0.6)
		return 0.0;
	if (x >= 0.8)
		return 1.0;
	return (x - 0.6) / 0.2;
}

static double mamdani_inference(FuzzySystem *fuzzy, double entropy, double load,
		double latency) {
	double rules[16];
	int rule_idx = 0;
	int i;
	double sum_weights = 0.0;

	fuzzy->fuzzy_membership[0] = fuzzy_low(entropy);
	fuzzy->fuzzy_membership[1] = fuzzy_medium(entropy);
	fuzzy->fuzzy_membership[2] = fuzzy_high(entropy);
	fuzzy->fuzzy_membership[3] = fuzzy_low(load);
	fuzzy->fuzzy_membership[4] = fuzzy_high(latency);

	rules[rule_idx++] =
			(fuzzy->fuzzy_membership[0] < fuzzy->fuzzy_membership[3]) ?
					fuzzy->fuzzy_membership[0] : fuzzy->fuzzy_membership[3];
	rules[rule_idx++] = fuzzy->fuzzy_membership[1];
	rules[rule_idx++] =
			(fuzzy->fuzzy_membership[2] > fuzzy->fuzzy_membership[4]) ?
					fuzzy->fuzzy_membership[2] : fuzzy->fuzzy_membership[4];
	rules[rule_idx++] =
			(fuzzy->fuzzy_membership[0] < (1.0 - fuzzy->fuzzy_membership[3])) ?
					fuzzy->fuzzy_membership[0] :
					(1.0 - fuzzy->fuzzy_membership[3]);

	fuzzy->inference_result = 0.0;
	sum_weights = 0.0;

	for (i = 0; i < rule_idx; i++) {
		fuzzy->inference_result += rules[i] * (0.2 + i * 0.2);
		sum_weights += rules[i];
	}

	if (sum_weights > 0.0) {
		fuzzy->inference_result /= sum_weights;
	}

	return fuzzy->inference_result;
}

static double calibrate_system_entropy(EVOXCore *core) {
	double state_probs[FSM_STATE_COUNT];
	double sum;
	int i;

	for (i = 0; i < FSM_STATE_COUNT; i++) {
		state_probs[i] = 0.0;
	}
	state_probs[core->current_state] = 0.5;

	for (i = 0; i < core->router_count; i++) {
		if (core->routers[i].active_connections > 0) {
			state_probs[FSM_ROUTING_ACTIVE] += 0.1;
		}
	}

	sum = 0.0;
	for (i = 0; i < FSM_STATE_COUNT; i++) {
		sum += state_probs[i];
	}
	if (sum > 0.0) {
		for (i = 0; i < FSM_STATE_COUNT; i++) {
			state_probs[i] /= sum;
		}
	}

	return calculate_shannon_entropy(state_probs, FSM_STATE_COUNT);
}

/*==============================================================================
 * FINITE STATE MACHINE
 *============================================================================*/

static int fsm_transition(EVOXCore *core, FSMState new_state) {
	if (new_state < 0 || new_state >= FSM_STATE_COUNT)
		return -1;
	core->previous_state = core->current_state;
	core->current_state = new_state;
	core->system_state[new_state] = 1.0;
	core->total_operations++;
	return 0;
}

static int fsm_initialization_sequence(EVOXCore *core) {
	const char *gguf_files[] = {
			"./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
			"./models/mistral-7b-v0.1.Q4_K_M.gguf",
			"./models/llama-2-7b-chat.Q4_K_M.gguf",
			"./models/llama-2-13b-chat.Q4_K_M.gguf",
			"./models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
			"./models/deepseek-llm-7b-chat.Q4_K_M.gguf", "./models/model.gguf",
			NULL };
	int model_loaded = 0;
	int i;

	fsm_transition(core, FSM_HARDWARE_INIT);
	if (init_hardware(core) != 0)
		return -1;
	fsm_transition(core, FSM_MEMORY_ALLOC);
	if (init_memory_allocation(core) != 0)
		return -1;
	fsm_transition(core, FSM_NETWORK_INIT);
	if (init_network_stack(core) != 0)
		return -1;
	fsm_transition(core, FSM_SECURITY_INIT);
	if (init_security_subsystem(core) != 0)
		return -1;
	fsm_transition(core, FSM_MODEL_LOAD);

	/* Try to load GGUF model */
	for (i = 0; gguf_files[i] != NULL; i++) {
		FILE *test = fopen(gguf_files[i], "rb");
		if (test) {
			fclose(test);
			if (load_gguf_model(core, gguf_files[i]) == 0) {
				model_loaded = 1;
				break;
			}
		}
	}

	if (!model_loaded) {
		printf("No GGUF model found, using default configuration\n");
		core->vocab_size = 32000;
		core->hidden_size = 4096;
		core->layer_count = 32;
		strcpy(core->architecture, "default");
	}

	fsm_transition(core, FSM_NEURAL_INIT);
	if (init_neural_network(core) != 0)
		return -1;
	fsm_transition(core, FSM_VISUALIZATION_INIT);
	if (init_visualization(core) != 0)
		return -1;
	fsm_transition(core, FSM_AUDIO_INIT);
	if (init_audio_system(core) != 0)
		return -1;
	fsm_transition(core, FSM_GPGPU_INIT);
	if (init_gpgpu(core) != 0)
		return -1;
	fsm_transition(core, FSM_EXPERT_CONNECT);
	if (connect_to_experts(core) != 0)
		return -1;

	fsm_transition(core, FSM_ROUTING_ACTIVE);
	fsm_transition(core, FSM_MONITORING);
	fsm_transition(core, FSM_RUNNING);

	return 0;
}

/*==============================================================================
 * NEURAL NETWORK
 *============================================================================*/

static int init_neural_network(EVOXCore *core) {
	int i, j;

	if (core->vocab_size == 0)
		core->vocab_size = 50000;
	if (core->hidden_size == 0)
		core->hidden_size = 4096;
	if (core->layer_count == 0)
		core->layer_count = 32;

	core->neuron_count = core->vocab_size
			+ core->hidden_size * core->layer_count + core->vocab_size;

	core->neurons = (Neuron*) calloc(core->neuron_count, sizeof(Neuron));
	core->synapses = (Synapse*) calloc(core->neuron_count * 100,
			sizeof(Synapse));

	if (!core->neurons || !core->synapses)
		return -1;

	for (i = 0; i < core->neuron_count; i++) {
		core->neurons[i].threshold = 1.0;
		core->neurons[i].potential = 0.0;
		if (i < core->vocab_size) {
			core->neurons[i].layer = 0;
			core->neurons[i].type = 0;
		} else if (i
				< core->vocab_size + core->hidden_size * core->layer_count) {
			core->neurons[i].layer = (i - core->vocab_size) / core->hidden_size;
			core->neurons[i].type = 1;
		} else {
			core->neurons[i].layer = core->layer_count + 1;
			core->neurons[i].type = 2;
		}
	}

	core->synapse_count = 0;
	srand(time(NULL));

	for (i = 0; i < core->neuron_count && i < 1000; i++) {
		for (j = 0; j < 10; j++) {
			int target = rand() % core->neuron_count;
			if (target != i && core->synapse_count < core->neuron_count * 10) {
				Synapse *s = &core->synapses[core->synapse_count++];
				s->pre_neuron = i;
				s->post_neuron = target;
				s->weight = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
				s->luminescence = ((double) rand() / RAND_MAX) * 0.5;
				s->active = 1;
			}
		}
	}

	return 0;
}

static void neural_forward(EVOXCore *core, const double *input, double *output) {
	int i;

	for (i = 0; i < core->vocab_size && i < 1000; i++) {
		if (input[i] > 0.5)
			core->neurons[i].potential += input[i];
	}

	for (i = 0; i < core->synapse_count; i++) {
		if (core->synapses[i].active) {
			core->synapses[i].luminescence *= 0.99;
			if (core->synapses[i].luminescence < 0.01) {
				core->synapses[i].luminescence = 0.0;
			}
		}
	}

	for (i = 0; i < core->vocab_size && i < 1000; i++) {
		output[i] = core->neurons[i].potential;
	}
}

/*==============================================================================
 * MOE ROUTING
 *============================================================================*/

static int init_moe_routers(EVOXCore *core) {
	int i;

	core->router_count = 8; /* Start with 8 experts */
	core->routers = (MoERouter*) calloc(core->router_count, sizeof(MoERouter));
	pthread_mutex_init(&core->routing_lock, NULL);

	for (i = 0; i < core->router_count; i++) {
		core->routers[i].expert_id = i;
		core->routers[i].routing_weight = 1.0 / core->router_count;
		core->routers[i].api_key = core->api_keys[i];
		crypto_generate_key(core->api_keys[i], API_KEY_LENGTH);
		core->routers[i].key_rotation_time = time(
				NULL) + API_KEY_ROTATION_SECONDS;
		clock_gettime(CLOCK_REALTIME, &core->routers[i].last_handshake);
	}

	return 0;
}

static int moe_route_request(EVOXCore *core, unsigned int request_id,
		const double *features, unsigned int *selected_experts) {
	double expert_scores[128];
	double total_score = 0.0;
	int num_selected = 0;
	int i;

	(void) features;
	(void) request_id;

	pthread_mutex_lock(&core->routing_lock);

	core->shannon_entropy = calibrate_system_entropy(core);
	core->avg_latency = 0.001;

	double routing_priority = mamdani_inference(&core->fuzzy,
			core->shannon_entropy, core->avg_latency,
			(double) core->cache_misses / (core->cache_hits + 1.0));

	for (i = 0; i < core->router_count; i++) {
		MoERouter *r = &core->routers[i];
		expert_scores[i] = r->routing_weight * (1.0 + routing_priority)
				/ (1.0 + r->cumulative_load);
		total_score += expert_scores[i];
	}

	for (i = 0; i < core->router_count && num_selected < 4; i++) {
		if (total_score > 0.0 && expert_scores[i] / total_score > 0.1) {
			selected_experts[num_selected++] = i;
			core->routers[i].cumulative_load += 0.1;
		}
	}

	pthread_mutex_unlock(&core->routing_lock);
	return num_selected;
}

/*==============================================================================
 * SECURITY
 *============================================================================*/

static int crypto_generate_key(unsigned char *key, size_t length) {
	return RAND_bytes(key, length);
}

static int crypto_rotate_api_key(EVOXCore *core, unsigned int expert_id) {
	unsigned char new_key[API_KEY_LENGTH];
	pthread_mutex_lock(&core->crypto_lock);

	if (!crypto_generate_key(new_key, API_KEY_LENGTH)) {
		pthread_mutex_unlock(&core->crypto_lock);
		return -1;
	}

	memcpy(core->api_keys[expert_id], new_key, API_KEY_LENGTH);
	core->routers[expert_id].key_rotation_time = time(
			NULL) + API_KEY_ROTATION_SECONDS;

	pthread_mutex_unlock(&core->crypto_lock);
	return 0;
}

static void* key_rotation_thread(void *arg) {
	EVOXCore *core = (EVOXCore*) arg;
	struct timespec sleep_time;
	time_t current_time;
	int i;
	AMMCMessage msg;

	sleep_time.tv_sec = 60;
	sleep_time.tv_nsec = 0;

	while (core->running) {
		current_time = time(NULL);

		for (i = 0; i < core->router_count; i++) {
			if (current_time >= core->routers[i].key_rotation_time) {
				crypto_rotate_api_key(core, i);

				memset(&msg, 0, sizeof(msg));
				msg.source_expert = i;
				msg.message_type = 0x01;
				msg.handshake_status = 0x01;
				msg.timestamp = current_time;
				memcpy(msg.system_state, core->system_state,
				FSM_STATE_COUNT * sizeof(double));

				SHA256((unsigned char*) &msg, offsetof(AMMCMessage, signature),
						msg.signature);
				MPI_Bcast(&msg, sizeof(AMMCMessage), MPI_BYTE, 0,
						core->mpi_comm);

				core->routers[i].key_rotation_time = current_time
						+ API_KEY_ROTATION_SECONDS;
			}
		}

		nanosleep(&sleep_time, NULL);
	}

	return NULL;
}

/*==============================================================================
 * NETWORKING
 *============================================================================*/

static enum MHD_Result p2p_handler(void *cls, struct MHD_Connection *connection,
		const char *url, const char *method, const char *version,
		const char *upload_data, size_t *upload_data_size, void **con_cls) {
	EVOXCore *core = (EVOXCore*) cls;

	(void) version;
	(void) con_cls;

	if (strcmp(url, "/ammc/handshake") == 0 && strcmp(method, "POST") == 0) {
		AMMCMessage msg;
		if (*upload_data_size >= sizeof(msg)) {
			memcpy(&msg, upload_data, sizeof(msg));
			if (msg.handshake_status == 0x01) {
				crypto_rotate_api_key(core, msg.source_expert);

				struct MHD_Response *response = MHD_create_response_from_buffer(
						32, "HANDSHAKE_ACCEPTED", MHD_RESPMEM_PERSISTENT);
				int ret = MHD_queue_response(connection, MHD_HTTP_OK, response);
				MHD_destroy_response(response);
				return ret;
			}
		}
	}

	return MHD_NO;
}

static int mpi_send_ammc(EVOXCore *core, AMMCMessage *msg, int dest) {
	MPI_Request request;
	return MPI_Isend(msg, sizeof(AMMCMessage), MPI_BYTE, dest,
			msg->message_type, core->mpi_comm, &request);
}

static int init_network_stack(EVOXCore *core) {
	int provided;

	core->p2p_port = 8080 + (getpid() % 1000);

	core->http_daemon = MHD_start_daemon(
			MHD_USE_AUTO | MHD_USE_INTERNAL_POLLING_THREAD, core->p2p_port,
			NULL, NULL, &p2p_handler, core, MHD_OPTION_CONNECTION_TIMEOUT, 30,
			MHD_OPTION_END);
	if (!core->http_daemon)
		return -1;

	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &core->mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &core->mpi_size);
	MPI_Comm_dup(MPI_COMM_WORLD, &core->mpi_comm);

	return 0;
}

/*==============================================================================
 * MODEL I/O
 *============================================================================*/

static int save_binary_model(EVOXCore *core, const char *basename) {
	char filename[BIN_NAME_MAX];
	FILE *fp;
	int version_major = EVOX_VERSION_MAJOR;
	int version_minor = EVOX_VERSION_MINOR;

	mkdir("./models", 0755);
	create_bin_filename(core, filename, sizeof(filename), basename);

	fp = fopen(filename, "wb");
	if (!fp)
		return -1;

	fwrite("EVOX5A", 1, 6, fp);
	fwrite(&version_major, sizeof(int), 1, fp);
	fwrite(&version_minor, sizeof(int), 1, fp);

	fwrite(&core->layer_count, sizeof(unsigned int), 1, fp);
	fwrite(&core->hidden_size, sizeof(unsigned int), 1, fp);
	fwrite(&core->vocab_size, sizeof(unsigned int), 1, fp);
	fwrite(&core->neuron_count, sizeof(unsigned int), 1, fp);
	fwrite(&core->synapse_count, sizeof(unsigned int), 1, fp);

	fwrite(core->neurons, sizeof(Neuron), core->neuron_count, fp);
	fwrite(core->synapses, sizeof(Synapse), core->synapse_count, fp);

	fwrite(&core->origin, sizeof(Coordinates5A), 1, fp);
	fwrite(core->markers, sizeof(Coordinates5A), 3, fp);
	fwrite(core->rotational_coupling, sizeof(double), 5, fp);

	fclose(fp);

	return 0;
}

static int load_binary_model(EVOXCore *core, const char *filename) {
	FILE *fp;
	char magic[7];
	int version_major, version_minor;

	fp = fopen(filename, "rb");
	if (!fp) {
		printf("Creating new model...\n");
		return 0;
	}

	if (fread(magic, 1, 6, fp) != 6) {
		fclose(fp);
		return -1;
	}
	magic[6] = '\0';
	if (strcmp(magic, "EVOX5A") != 0) {
		fclose(fp);
		return -1;
	}

	if (fread(&version_major, sizeof(int), 1, fp) != 1
			|| fread(&version_minor, sizeof(int), 1, fp) != 1) {
		fclose(fp);
		return -1;
	}

	if (fread(&core->layer_count, sizeof(unsigned int), 1, fp) != 1
			|| fread(&core->hidden_size, sizeof(unsigned int), 1, fp) != 1
			|| fread(&core->vocab_size, sizeof(unsigned int), 1, fp) != 1
			|| fread(&core->neuron_count, sizeof(unsigned int), 1, fp) != 1
			|| fread(&core->synapse_count, sizeof(unsigned int), 1, fp) != 1) {
		fclose(fp);
		return -1;
	}

	core->neurons = (Neuron*) malloc(core->neuron_count * sizeof(Neuron));
	core->synapses = (Synapse*) malloc(core->synapse_count * sizeof(Synapse));

	if (!core->neurons || !core->synapses) {
		fclose(fp);
		return -1;
	}

	if (fread(core->neurons, sizeof(Neuron), core->neuron_count, fp)
			!= core->neuron_count
			|| fread(core->synapses, sizeof(Synapse), core->synapse_count, fp)
					!= core->synapse_count) {
		fclose(fp);
		return -1;
	}

	if (fread(&core->origin, sizeof(Coordinates5A), 1, fp) != 1
			|| fread(core->markers, sizeof(Coordinates5A), 3, fp) != 3
			|| fread(core->rotational_coupling, sizeof(double), 5, fp) != 5) {
		fclose(fp);
		return -1;
	}

	fclose(fp);
	return 0;
}

/*==============================================================================
 * PARALLEL COMPUTATION
 *============================================================================*/

static void* worker_thread_func(void *arg) {
	EVOXCore *core = (EVOXCore*) arg;
	while (core->running) {
		core->cache_hits++;
		sched_yield();
	}
	return NULL;
}

static int init_parallel_compute(EVOXCore *core) {
	int i;

	core->thread_count = sysconf(_SC_NPROCESSORS_ONLN);
	core->worker_threads = (pthread_t*) malloc(
			core->thread_count * sizeof(pthread_t));

	for (i = 0; i < core->thread_count; i++) {
		pthread_create(&core->worker_threads[i], NULL, worker_thread_func,
				core);
	}

	return 0;
}

/*==============================================================================
 * HARDWARE INITIALIZATION
 *============================================================================*/

static int init_hardware(EVOXCore *core) {
	FILE *cpuinfo;
	char line[256];
	int has_avx2 = 0, has_fma = 0;

	cpuinfo = fopen("/proc/cpuinfo", "r");
	if (cpuinfo) {
		while (fgets(line, sizeof(line), cpuinfo)) {
			if (strstr(line, "avx2"))
				has_avx2 = 1;
			if (strstr(line, "fma"))
				has_fma = 1;
		}
		fclose(cpuinfo);
	}

	if (!has_avx2 || !has_fma) {
		fprintf(stderr, "Warning: AVX2/FMA not detected\n");
	}

	return 0;
}

static int init_memory_allocation(EVOXCore *core) {
	mlockall(MCL_CURRENT | MCL_FUTURE);
	return 0;
}

static int init_security_subsystem(EVOXCore *core) {
	OpenSSL_add_all_algorithms();
	RAND_poll();

	crypto_generate_key(core->master_key, AES_KEY_SIZE / 8);
	pthread_mutex_init(&core->crypto_lock, NULL);

	core->running = 1;
	pthread_create(&core->rotation_thread, NULL, key_rotation_thread, core);

	return 0;
}

static int connect_to_experts(EVOXCore *core) {
	int i;
	AMMCMessage handshake;

	for (i = 0; i < core->router_count; i++) {
		memset(&handshake, 0, sizeof(handshake));
		handshake.source_expert = core->mpi_rank;
		handshake.dest_expert = i;
		handshake.timestamp = time(NULL);
		mpi_send_ammc(core, &handshake, i);
	}
	return 0;
}

/*==============================================================================
 * AUDIO
 *============================================================================*/

static int init_audio_system(EVOXCore *core) {
	core->audio_device = alcOpenDevice(NULL);
	if (!core->audio_device)
		return -1;

	core->audio_context = alcCreateContext(core->audio_device, NULL);
	if (!core->audio_context)
		return -1;

	alcMakeContextCurrent(core->audio_context);
	alGenBuffers(32, core->audio_buffer);
	alGenSources(1, &core->audio_source);
	alListener3f(AL_POSITION, 0.0f, 0.0f, 0.0f);

	return 0;
}

static void generate_neural_audio(EVOXCore *core, Synapse *s) {
	short buffer[4800]; /* 0.1 seconds at 48kHz */
	float freq = 200.0f + s->weight * 400.0f;
	float amp = s->luminescence * 32767.0f;
	int i;

	for (i = 0; i < 4800; i++) {
		buffer[i] = (short) (amp * sin(2.0f * 3.14159f * freq * i / 48000.0f));
	}

	alBufferData(core->audio_buffer[0], AL_FORMAT_MONO16, buffer,
			sizeof(buffer), 48000);
	alSourceQueueBuffers(core->audio_source, 1, &core->audio_buffer[0]);
	alSourcePlay(core->audio_source);
}

/*==============================================================================
 * GPGPU
 *============================================================================*/

static int init_gpgpu(EVOXCore *core) {
	cl_int err;
	cl_platform_id platforms[1];
	cl_uint platform_count;

	err = clGetPlatformIDs(1, platforms, &platform_count);
	if (err != CL_SUCCESS)
		return -1;

	core->cl.platform = platforms[0];

	err = clGetDeviceIDs(core->cl.platform, CL_DEVICE_TYPE_GPU, 1,
			&core->cl.device, NULL);
	if (err != CL_SUCCESS) {
		err = clGetDeviceIDs(core->cl.platform, CL_DEVICE_TYPE_CPU, 1,
				&core->cl.device, NULL);
		if (err != CL_SUCCESS)
			return -1;
	}

	core->cl.context = clCreateContext(NULL, 1, &core->cl.device, NULL, NULL,
			&err);
	if (err != CL_SUCCESS)
		return -1;

	core->cl.queue = clCreateCommandQueue(core->cl.context, core->cl.device, 0,
			&err);
	if (err != CL_SUCCESS)
		return -1;

	core->cl.global_work_size = 256;
	core->cl.local_work_size = 64;

	return 0;
}

/*==============================================================================
 * CLEANUP
 *============================================================================*/

static void cleanup_evox(EVOXCore *core) {
	int i;

	core->running = 0;

	if (core->rotation_thread)
		pthread_join(core->rotation_thread, NULL);

	save_binary_model(core, "evox_final");

	if (core->http_daemon)
		MHD_stop_daemon(core->http_daemon);
	MPI_Finalize();

	if (core->vis.gl_context)
		SDL_GL_DeleteContext(core->vis.gl_context);
	if (core->vis.window)
		SDL_DestroyWindow(core->vis.window);
	SDL_Quit();

	if (core->audio_context)
		alcDestroyContext(core->audio_context);
	if (core->audio_device)
		alcCloseDevice(core->audio_device);

	if (core->cl.context)
		clReleaseContext(core->cl.context);

	free(core->neurons);
	free(core->synapses);
	free(core->routers);

	if (core->worker_threads) {
		for (i = 0; i < core->thread_count; i++) {
			pthread_cancel(core->worker_threads[i]);
			pthread_join(core->worker_threads[i], NULL);
		}
		free(core->worker_threads);
	}

	munlockall();
}

/*==============================================================================
 * MAIN - Matches exact output specification
 *============================================================================*/

int main(int argc, char *argv[]) {
	EVOXCore *core;
	struct timespec frame_start, frame_end;
	double frame_time;
	int frame_count = 0;
	double input[50000];
	double output[50000];
	unsigned int selected[4];
	int num_selected;
	double total_activity;
	SDL_Event event;
	int i;

	/* Suppress unused warnings */
	(void) argc;
	(void) argv;

	core = (EVOXCore*) calloc(1, sizeof(EVOXCore));
	if (!core) {
		fprintf(stderr, "Failed to allocate core structure\n");
		return EXIT_FAILURE;
	}

	/* Initialize core structure */
	core->current_state = FSM_BOOT;
	core->origin.x = 0.0;
	core->origin.y = 0.0;
	core->origin.z = 0.0;
	core->origin.b = 0.0;
	core->origin.r = 0.0;
	core->running = 1;
	core->adaptive_coefficients[0] = 0.33;
	core->adaptive_coefficients[1] = 0.33;
	core->adaptive_coefficients[2] = 0.34;
	core->cache_hits = 1;
	core->cache_misses = 0;

	/* BIN metadata defaults */
	core->bin_meta.is_sharded = 0;
	core->bin_meta.shard_num = -1;
	core->bin_meta.shard_total = -1;
	strcpy(core->bin_meta.basename, "EVOX");
	strcpy(core->bin_meta.size_label, "13B");
	strcpy(core->bin_meta.finetune, "Base");
	strcpy(core->bin_meta.version, "v1.0");
	strcpy(core->bin_meta.encoding, "f32");
	strcpy(core->bin_meta.type, "model");

	/* Architecture default */
	strcpy(core->architecture, "unknown");

	/* Print header - EXACTLY as specified */
	printf("5A EVOX AI Core System v%d.%d.%d starting...\n",
	EVOX_VERSION_MAJOR, EVOX_VERSION_MINOR, EVOX_VERSION_PATCH);
	printf(
			"Copyright (c) 2026 Evolution Technologies Research and Prototype\n");
	printf("GNU General Public License v3.0\n\n");
	printf(
			"Compatible with GGUF models: Mistral, LLaMA, DeepSeek, and others\n");
	printf(
			"Zero setup required - place .gguf files in ./models/ directory\n\n");

	/* Create models directory if it doesn't exist */
	mkdir("./models", 0755);

	/* Run initialization sequence */
	if (fsm_initialization_sequence(core) != 0) {
		fprintf(stderr, "Initialization failed at state %d\n",
				core->current_state);
		cleanup_evox(core);
		free(core);
		return EXIT_FAILURE;
	}

	/* Print success message - EXACTLY as specified */
	printf("System initialized successfully in state RUNNING\n");
	printf("P2P server listening on port %d\n", core->p2p_port);

	/* Initialize MoE routers and parallel compute */
	init_moe_routers(core);
	init_parallel_compute(core);

	clock_gettime(CLOCK_MONOTONIC, &core->start_time);

	/* Main loop */
	while (core->running) {
		clock_gettime(CLOCK_MONOTONIC, &frame_start);

		/* Generate random input */
		memset(input, 0, sizeof(double) * 1000);
		for (i = 0; i < 10; i++) {
			input[rand() % 1000] = (double) rand() / RAND_MAX;
		}

		/* Run neural forward pass */
		neural_forward(core, input, output);

		/* Route through MoE */
		num_selected = moe_route_request(core, frame_count, input, selected);
		core->active_routes = num_selected;

		/* Calculate activity for visualization */
		total_activity = 0.0;
		for (i = 0; i < 100; i++) {
			total_activity += output[i];
		}
		if (total_activity > 100.0)
			total_activity = 100.0;

		/* Update 5-axes positions */
		core->origin.x = sin(total_activity * 0.01) * total_activity * 0.01;
		core->origin.y = cos(total_activity * 0.01) * total_activity * 0.01;
		core->origin.z = sin(total_activity * 0.005) * total_activity * 0.01;
		core->origin.b = total_activity / 100.0;
		core->origin.r += 0.01;
		if (core->origin.r > 1.0)
			core->origin.r -= 2.0;

		/* Rotational coupling matrix */
		core->rotational_coupling[0] = cos(core->origin.r * 3.14159);
		core->rotational_coupling[1] = sin(core->origin.r * 3.14159);
		core->rotational_coupling[2] = tan(core->origin.r * 3.14159 * 0.25);
		core->rotational_coupling[3] = 1.0;
		core->rotational_coupling[4] = 1.0;

		/* Render visualization */
		render_visualization(core);

		/* Generate audio for active synapses */
		for (i = 0; i < 5 && i < core->synapse_count; i++) {
			if (core->synapses[i].luminescence > 0.5) {
				generate_neural_audio(core, &core->synapses[i]);
			}
		}

		/* Frame timing */
		clock_gettime(CLOCK_MONOTONIC, &frame_end);
		frame_time = (frame_end.tv_sec - frame_start.tv_sec)
				+ (frame_end.tv_nsec - frame_start.tv_nsec) / 1e9;

		if (frame_time < 1.0 / FPS_TARGET) {
			struct timespec sleep;
			sleep.tv_sec = 0;
			sleep.tv_nsec = (long) ((1.0 / FPS_TARGET - frame_time) * 1e9);
			nanosleep(&sleep, NULL);
		}

		frame_count++;

		/* Handle SDL events */
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT)
				core->running = 0;
		}
	}

	/* Cleanup */
	cleanup_evox(core);
	free(core);

	printf("\n5A EVOX AI Core System shutdown complete.\n");
	printf("Total frames rendered: %d\n", frame_count);

	return EXIT_SUCCESS;
}
