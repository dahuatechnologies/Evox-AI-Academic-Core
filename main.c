/**
 * EVOX AI CORE V 0.1 - 3D INTERACTIVE MULTIMEDIA PREVIEW
 * ============================================================
 * Copyright (c) 2026 Evolution Technologies Research and Prototype - All Rights Reserved
 *
 *
 * FEATURES:
 * • Starts at origin (0,0,0) with incremental node generation
 * • RGB axes with smooth rotation in R (radians)
 * • One-by-one neuron and synapse creation animation
 * • Real-time multimedia feedback (visual + audio)
 * • Interactive camera controls
 *
 * COMPILATION: gcc -std=c90 -D_GNU_SOURCE -pthread -o evox main.c \
 *              -lSDL2 -lGL -lGLU -lglut -lopenal -lm -lrt
 *
 * RUN: ./evox
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <execinfo.h>
#include <float.h>

/* External Libraries */
#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <AL/al.h>
#include <AL/alc.h>

/*-----------------------------------------------------------------------------
 * SAFETY MACROS
 *----------------------------------------------------------------------------*/

#define SAFE_CHECK(ptr) do { \
    if (!(ptr)) { \
        fprintf(stderr, "[FATAL] NULL pointer at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define SAFE_FREE(ptr) do { \
    if (ptr) { \
        free(ptr); \
        ptr = NULL; \
    } \
} while(0)

#define BOUNDS_CHECK(idx, max) do { \
    if ((idx) < 0 || (idx) >= (max)) { \
        fprintf(stderr, "[FATAL] Index %d out of bounds [0,%d) at %s:%d\n", \
                (int)(idx), (int)(max), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(x,min,max) (MIN(MAX((x), (min)), (max)))

/*-----------------------------------------------------------------------------
 * CONSTANTS
 *----------------------------------------------------------------------------*/

#define EVOX_VERSION            "13.0.0"
#define EVOX_CODENAME           "Neural Sentinel"

/* Window dimensions */
#define WINDOW_WIDTH            1280
#define WINDOW_HEIGHT           720

/* Hypergraph - Incremental build */
#define MAX_NODES               512
#define MAX_EDGES               10000
#define MAX_EXPERTS             8
#define SYNAPSE_DENSITY         0.03f
#define MAX_EDGES_PER_NODE      20
#define BUILD_SPEED             5       /* Nodes per frame */

/* Audio */
#define MAX_AUDIO_SOURCES       4
#define AUDIO_SAMPLE_RATE       44100
#define AUDIO_BUFFER_SIZE       4096
#define AUDIO_MIN_FREQ          200.0f
#define AUDIO_MAX_FREQ          600.0f
#define AUDIO_MAX_VOLUME        0.2f

/* Visualization */
#define AXIS_LENGTH             150.0f
#define GRID_SIZE               300.0f
#define GRID_STEP               50.0f
#define ORIGIN_SIZE             5.0f
#define ROTATION_SPEED          0.02f    /* Radians per frame */

/*-----------------------------------------------------------------------------
 * BGRA COLOR STRUCTURE
 *----------------------------------------------------------------------------*/

typedef struct {
	uint8_t b;
	uint8_t g;
	uint8_t r;
	uint8_t a;
} ColorBGRA;

ColorBGRA color_white(void) {
	ColorBGRA c = { 255, 255, 255, 255 };
	return c;
}

ColorBGRA color_red(void) {
	ColorBGRA c = { 0, 0, 255, 255 };
	return c;
}

ColorBGRA color_green(void) {
	ColorBGRA c = { 0, 255, 0, 255 };
	return c;
}

ColorBGRA color_blue(void) {
	ColorBGRA c = { 255, 0, 0, 255 };
	return c;
}

ColorBGRA color_yellow(void) {
	ColorBGRA c = { 0, 255, 255, 255 };
	return c;
}

ColorBGRA color_cyan(void) {
	ColorBGRA c = { 255, 255, 0, 255 };
	return c;
}

ColorBGRA color_magenta(void) {
	ColorBGRA c = { 255, 0, 255, 255 };
	return c;
}

ColorBGRA color_heatmap(float value, float min, float max) {
	float norm;
	uint8_t r, g, b;
	ColorBGRA result;

	if (value < min)
		value = min;
	if (value > max)
		value = max;

	norm = (value - min) / (max - min);

	if (norm < 0.33f) {
		r = 0;
		g = (uint8_t) (255 * norm * 3);
		b = 255;
	} else if (norm < 0.66f) {
		r = (uint8_t) (255 * (norm - 0.33f) * 3);
		g = 255;
		b = (uint8_t) (255 * (1 - (norm - 0.33f) * 3));
	} else {
		r = 255;
		g = (uint8_t) (255 * (1 - (norm - 0.66f) * 3));
		b = 0;
	}

	result.b = b;
	result.g = g;
	result.r = r;
	result.a = 255;
	return result;
}

/*-----------------------------------------------------------------------------
 * UTILITY FUNCTIONS
 *----------------------------------------------------------------------------*/

float random_float(float min, float max) {
	return min + ((float) rand() / RAND_MAX) * (max - min);
}

float sigmoid(float x) {
	if (x > 10.0f)
		return 1.0f;
	if (x < -10.0f)
		return 0.0f;
	return 1.0f / (1.0f + expf(-x));
}

uint64_t get_timestamp_ms(void) {
	struct timeval tv;
	if (gettimeofday(&tv, NULL) == 0) {
		return (uint64_t) tv.tv_sec * 1000 + (uint64_t) tv.tv_usec / 1000;
	}
	return 0;
}

/*-----------------------------------------------------------------------------
 * OPENAL 3D AUDIO
 *----------------------------------------------------------------------------*/

typedef struct {
	ALCdevice *device;
	ALCcontext *context;
	ALuint sources[MAX_AUDIO_SOURCES];
	ALuint buffers[MAX_AUDIO_SOURCES];
	float source_positions[MAX_AUDIO_SOURCES][3];
	int source_active[MAX_AUDIO_SOURCES];
	pthread_mutex_t audio_mutex;
	int initialized;
} AudioContext;

void generate_sine_wave(short *buffer, int samples, float frequency,
		int sample_rate) {
	int i;
	float amplitude = 8000.0f;
	float phase = 0.0f;
	float phase_step = 2.0f * (float) M_PI * frequency / sample_rate;

	for (i = 0; i < samples; i++) {
		buffer[i] = (short) (amplitude * sinf(phase));
		phase += phase_step;
		if (phase > 2.0f * (float) M_PI)
			phase -= 2.0f * (float) M_PI;
	}
}

int init_audio(AudioContext *audio) {
	int i;
	short silent_data[AUDIO_BUFFER_SIZE];

	SAFE_CHECK(audio);

	printf("[AUDIO] Initializing OpenAL...\n");

	memset(audio, 0, sizeof(AudioContext));
	pthread_mutex_init(&audio->audio_mutex, NULL);

	memset(silent_data, 0, sizeof(silent_data));

	audio->device = alcOpenDevice(NULL);
	if (!audio->device) {
		fprintf(stderr, "[AUDIO] Failed to open audio device\n");
		return -1;
	}

	audio->context = alcCreateContext(audio->device, NULL);
	if (!audio->context) {
		fprintf(stderr, "[AUDIO] Failed to create audio context\n");
		alcCloseDevice(audio->device);
		return -1;
	}

	if (!alcMakeContextCurrent(audio->context)) {
		fprintf(stderr, "[AUDIO] Failed to make context current\n");
		alcDestroyContext(audio->context);
		alcCloseDevice(audio->device);
		return -1;
	}

	alGetError();

	alGenSources(MAX_AUDIO_SOURCES, audio->sources);
	alGenBuffers(MAX_AUDIO_SOURCES, audio->buffers);

	for (i = 0; i < MAX_AUDIO_SOURCES; i++) {
		alBufferData(audio->buffers[i], AL_FORMAT_MONO16, silent_data,
				AUDIO_BUFFER_SIZE * sizeof(short),
				AUDIO_SAMPLE_RATE);

		alSourcei(audio->sources[i], AL_BUFFER, audio->buffers[i]);
		alSourcef(audio->sources[i], AL_GAIN, 0.0f);
		alSource3f(audio->sources[i], AL_POSITION, 0, 0, 0);
		alSourcei(audio->sources[i], AL_LOOPING, AL_TRUE);

		audio->source_active[i] = 0;
	}

	alListener3f(AL_POSITION, 0, 0, 0);

	audio->initialized = 1;

	printf("[AUDIO] Initialized successfully\n");
	return 0;
}

void play_audio_note(AudioContext *audio, int source_idx, float frequency,
		float volume, float x, float y, float z) {
	short data[AUDIO_BUFFER_SIZE];

	if (!audio || !audio->initialized)
		return;
	if (source_idx < 0 || source_idx >= MAX_AUDIO_SOURCES)
		return;

	pthread_mutex_lock(&audio->audio_mutex);

	frequency = CLAMP(frequency, AUDIO_MIN_FREQ, AUDIO_MAX_FREQ);
	volume = CLAMP(volume, 0.05f, AUDIO_MAX_VOLUME);

	generate_sine_wave(data, AUDIO_BUFFER_SIZE, frequency, AUDIO_SAMPLE_RATE);

	alBufferData(audio->buffers[source_idx], AL_FORMAT_MONO16, data,
			AUDIO_BUFFER_SIZE * sizeof(short),
			AUDIO_SAMPLE_RATE);

	alSourcef(audio->sources[source_idx], AL_GAIN, volume);
	alSource3f(audio->sources[source_idx], AL_POSITION, x, y, z);

	alSourcePlay(audio->sources[source_idx]);

	audio->source_active[source_idx] = 1;
	audio->source_positions[source_idx][0] = x;
	audio->source_positions[source_idx][1] = y;
	audio->source_positions[source_idx][2] = z;

	pthread_mutex_unlock(&audio->audio_mutex);
}

void stop_audio_source(AudioContext *audio, int source_idx) {
	if (!audio || !audio->initialized)
		return;
	if (source_idx < 0 || source_idx >= MAX_AUDIO_SOURCES)
		return;

	pthread_mutex_lock(&audio->audio_mutex);
	alSourceStop(audio->sources[source_idx]);
	alSourcef(audio->sources[source_idx], AL_GAIN, 0.0f);
	audio->source_active[source_idx] = 0;
	pthread_mutex_unlock(&audio->audio_mutex);
}

void cleanup_audio(AudioContext *audio) {
	int i;

	if (!audio)
		return;

	printf("[AUDIO] Cleaning up...\n");

	pthread_mutex_lock(&audio->audio_mutex);

	if (audio->initialized) {
		for (i = 0; i < MAX_AUDIO_SOURCES; i++) {
			alSourceStop(audio->sources[i]);
		}

		alDeleteSources(MAX_AUDIO_SOURCES, audio->sources);
		alDeleteBuffers(MAX_AUDIO_SOURCES, audio->buffers);

		alcMakeContextCurrent(NULL);
		if (audio->context)
			alcDestroyContext(audio->context);
		if (audio->device)
			alcCloseDevice(audio->device);

		audio->initialized = 0;
	}

	pthread_mutex_unlock(&audio->audio_mutex);
	pthread_mutex_destroy(&audio->audio_mutex);

	printf("[AUDIO] Cleanup complete\n");
}

/*-----------------------------------------------------------------------------
 * HYPERGRAPH - INCREMENTAL BUILD FROM ORIGIN
 *----------------------------------------------------------------------------*/

typedef struct {
	float x, y, z;
	float activation;
	float potential;
	unsigned int expert_id;
	unsigned int firing_count;
	float last_fired;
	unsigned int edge_indices[MAX_EDGES_PER_NODE];
	unsigned int edge_count;
	ColorBGRA base_color;
	ColorBGRA active_color;
	int visible; /* Whether node has been built */
	int initialized;
} HypergraphNode;

typedef struct {
	unsigned int source;
	unsigned int target;
	float strength;
	float frequency;
	float luminescence;
	ColorBGRA color;
	unsigned int active;
	unsigned int packet_count;
	int visible; /* Whether edge has been built */
	int initialized;
} HypergraphEdge;

typedef struct {
	HypergraphNode nodes[MAX_NODES];
	HypergraphEdge edges[MAX_EDGES];
	unsigned int total_nodes;
	unsigned int total_edges;
	unsigned int visible_nodes;
	unsigned int visible_edges;
	float global_entropy;

	AudioContext *audio;
	int audio_counter;
	int build_phase; /* Current build phase (0=origin, 1=nodes, 2=edges) */
	float build_progress; /* 0.0 to 1.0 */

	int initialized;
} Hypergraph;

void init_hypergraph(Hypergraph *hg, AudioContext *audio) {
	int i, j;
	float theta, phi, r;
	unsigned int edge_idx;

	SAFE_CHECK(hg);

	printf("[HYPERGRAPH] Initializing...\n");

	memset(hg, 0, sizeof(Hypergraph));

	hg->audio = audio;
	hg->audio_counter = 0;
	hg->total_nodes = MAX_NODES;
	hg->total_edges = 0;
	hg->visible_nodes = 0;
	hg->visible_edges = 0;
	hg->build_phase = 0; /* Start at origin */
	hg->build_progress = 0.0f;

	/* Initialize all nodes (but not visible yet) */
	for (i = 0; i < MAX_NODES; i++) {
		theta = 2.0f * (float) M_PI * i / MAX_NODES;
		phi = acosf(2.0f * i / MAX_NODES - 1.0f);
		r = 150.0f + 30.0f * sinf(i * 0.1f);

		hg->nodes[i].x = r * sinf(phi) * cosf(theta);
		hg->nodes[i].y = r * sinf(phi) * sinf(theta) * 0.7f;
		hg->nodes[i].z = r * cosf(phi);
		hg->nodes[i].activation = random_float(0.0f, 0.3f);
		hg->nodes[i].potential = 0.0f;
		hg->nodes[i].expert_id = i % MAX_EXPERTS;
		hg->nodes[i].firing_count = 0;
		hg->nodes[i].last_fired = 0.0f;
		hg->nodes[i].edge_count = 0;
		hg->nodes[i].visible = 0; /* Not visible yet */

		switch (hg->nodes[i].expert_id % 4) {
		case 0:
			hg->nodes[i].base_color = color_red();
			hg->nodes[i].active_color = color_yellow();
			break;
		case 1:
			hg->nodes[i].base_color = color_green();
			hg->nodes[i].active_color = color_cyan();
			break;
		case 2:
			hg->nodes[i].base_color = color_blue();
			hg->nodes[i].active_color = color_magenta();
			break;
		default:
			hg->nodes[i].base_color = color_white();
			hg->nodes[i].active_color = color_yellow();
			break;
		}

		hg->nodes[i].initialized = 1;
	}

	/* Create all edges (but not visible yet) */
	for (i = 0; i < MAX_NODES; i++) {
		for (j = i + 1; j < MAX_NODES && j < i + 20; j++) {
			if (random_float(0, 1) < SYNAPSE_DENSITY
					&& hg->total_edges < MAX_EDGES) {
				edge_idx = hg->total_edges++;

				hg->edges[edge_idx].source = i;
				hg->edges[edge_idx].target = j;
				hg->edges[edge_idx].strength = random_float(0.3f, 1.0f);
				hg->edges[edge_idx].frequency = random_float(0.5f, 2.0f);
				hg->edges[edge_idx].luminescence = 0.0f;
				hg->edges[edge_idx].color = color_heatmap(
						hg->edges[edge_idx].strength, 0, 1);
				hg->edges[edge_idx].active = 1;
				hg->edges[edge_idx].packet_count = 0;
				hg->edges[edge_idx].visible = 0; /* Not visible yet */
				hg->edges[edge_idx].initialized = 1;
			}
		}
	}

	hg->initialized = 1;

	printf("[HYPERGRAPH] Created %d edges total\n", hg->total_edges);
}

void update_hypergraph_build(Hypergraph *hg) {
	int i, e;
	int built = 0;
	static int frame = 0;

	if (!hg || !hg->initialized)
		return;

	frame++;
	hg->audio_counter++;

	/* Phase 0: Origin only (first 30 frames) */
	if (hg->build_phase == 0) {
		hg->build_progress += 0.02f;
		if (hg->build_progress >= 1.0f || frame > 60) {
			hg->build_phase = 1;
			hg->build_progress = 0.0f;

			/* Play audio for origin completion */
			if (hg->audio && hg->audio->initialized) {
				play_audio_note(hg->audio, 0, 440.0f, 0.2f, 0, 0, 0);
			}
		}
		return;
	}

	/* Phase 1: Build nodes one by one */
	if (hg->build_phase == 1) {
		/* Add BUILD_SPEED nodes per frame */
		for (i = 0; i < BUILD_SPEED && hg->visible_nodes < hg->total_nodes;
				i++) {
			int node_idx = hg->visible_nodes;
			hg->nodes[node_idx].visible = 1;
			hg->visible_nodes++;
			built++;

			/* Play audio for each new node */
			if (hg->audio && hg->audio->initialized && built % 5 == 0) {
				float freq = 300.0f + (node_idx % 5) * 50.0f;
				play_audio_note(hg->audio, 1, freq, 0.1f,
						hg->nodes[node_idx].x * 0.01f,
						hg->nodes[node_idx].y * 0.01f,
						hg->nodes[node_idx].z * 0.01f);
			}
		}

		hg->build_progress = (float) hg->visible_nodes / hg->total_nodes;

		/* When all nodes built, move to edges */
		if (hg->visible_nodes >= hg->total_nodes) {
			hg->build_phase = 2;
			hg->build_progress = 0.0f;

			/* Play audio for completion */
			if (hg->audio && hg->audio->initialized) {
				play_audio_note(hg->audio, 2, 523.0f, 0.3f, 0, 0, 0);
			}
		}
		return;
	}

	/* Phase 2: Build edges one by one */
	if (hg->build_phase == 2) {
		/* Add edges faster */
		for (i = 0; i < BUILD_SPEED * 2 && hg->visible_edges < hg->total_edges;
				i++) {
			hg->edges[hg->visible_edges].visible = 1;
			hg->visible_edges++;
			built++;
		}

		hg->build_progress = (float) hg->visible_edges / hg->total_edges;

		/* Play occasional audio during edge building */
		if (hg->audio && hg->audio->initialized && built > 0
				&& hg->visible_edges % 20 == 0) {
			play_audio_note(hg->audio, 3,
					400.0f + (hg->visible_edges % 10) * 20.0f, 0.15f, 0, 0, 0);
		}
	}
}

void update_hypergraph_activity(Hypergraph *hg) {
	int i, e;
	unsigned int src, tgt;
	float signal;

	if (!hg || !hg->initialized)
		return;
	if (hg->visible_nodes == 0)
		return;

	/* Decay potentials for visible nodes */
	for (i = 0; i < hg->visible_nodes; i++) {
		hg->nodes[i].potential *= 0.96f;
	}

	/* Process visible edges */
	for (e = 0; e < hg->visible_edges; e++) {
		if (hg->edges[e].active) {
			src = hg->edges[e].source;
			tgt = hg->edges[e].target;

			if (src < hg->visible_nodes && tgt < hg->visible_nodes) {
				signal = hg->nodes[src].activation * hg->edges[e].strength;
				hg->nodes[tgt].potential += signal * 0.15f;

				hg->edges[e].luminescence = signal;
				hg->edges[e].packet_count++;
			}
		}
	}

	/* Fire visible neurons */
	for (i = 0; i < hg->visible_nodes; i++) {
		if (hg->nodes[i].potential > 1.0f) {
			hg->nodes[i].activation = 1.0f;
			hg->nodes[i].firing_count++;
			hg->nodes[i].last_fired = get_timestamp_ms() / 1000.0f;
			hg->nodes[i].potential = 0.0f;
		} else {
			hg->nodes[i].activation = 0.7f
					* sigmoid(hg->nodes[i].potential * 4.0f - 2.0f)
					+ 0.3f * hg->nodes[i].activation;
		}
	}

	/* Simple entropy */
	float sum = 0.0f;
	for (i = 0; i < hg->visible_nodes && i < 100; i++) {
		sum += hg->nodes[i].activation;
	}
	hg->global_entropy =
			(hg->visible_nodes > 0) ? sum / hg->visible_nodes : 0.0f;
}

/*-----------------------------------------------------------------------------
 * OPENGL RENDERING - WITH ORIGIN FOCUS
 *----------------------------------------------------------------------------*/

typedef struct {
	SDL_Window *window;
	SDL_GLContext gl_context;
	int width;
	int height;
	int fullscreen;
	float camera_angle;
	float camera_elevation;
	float camera_distance;
	float rotation_r; /* Rotation in radians for axes */
	int mouse_x, mouse_y;
	int mouse_buttons[3];
	int initialized;
} OpenGLContext;

void init_opengl(OpenGLContext *gl, int width, int height) {
	SAFE_CHECK(gl);

	printf("[OPENGL] Initializing...\n");

	memset(gl, 0, sizeof(OpenGLContext));

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		fprintf(stderr, "[SDL] Failed: %s\n", SDL_GetError());
		return;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	gl->window = SDL_CreateWindow("EVOX AI CORE - 3D Interactive Preview",
	SDL_WINDOWPOS_CENTERED,
	SDL_WINDOWPOS_CENTERED, width, height,
			SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

	if (!gl->window) {
		fprintf(stderr, "[SDL] Window creation failed\n");
		SDL_Quit();
		return;
	}

	gl->gl_context = SDL_GL_CreateContext(gl->window);
	if (!gl->gl_context) {
		fprintf(stderr, "[OPENGL] Context creation failed\n");
		SDL_DestroyWindow(gl->window);
		SDL_Quit();
		return;
	}

	SDL_GL_SetSwapInterval(1);

	gl->width = width;
	gl->height = height;
	gl->camera_angle = 45.0f;
	gl->camera_elevation = 30.0f;
	gl->camera_distance = 400.0f;
	gl->rotation_r = 0.0f;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double) width / height, 1.0, 1000.0);
	glMatrixMode(GL_MODELVIEW);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_LINE_SMOOTH);

	glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

	gl->initialized = 1;

	printf("[OPENGL] Initialized\n");
}

void draw_origin(void) {
	/* Draw a bright sphere at the origin */
	glPushMatrix();
	glTranslatef(0, 0, 0);

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glutSolidSphere(ORIGIN_SIZE, 16, 16);

	/* Add glow effect */
	glColor4f(1.0f, 1.0f, 0.5f, 0.3f);
	glutSolidSphere(ORIGIN_SIZE * 2, 16, 16);

	glPopMatrix();
}

void draw_axes_rotating(float r) {
	glLineWidth(3.0f);

	/* Save current matrix */
	glPushMatrix();

	/* Apply rotation in R (radians) */
	glRotatef(r * 180.0f / (float) M_PI, 1.0f, 1.0f, 0.0f);

	/* X axis - Red */
	glBegin(GL_LINES);
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glVertex3f(0, 0, 0);
	glVertex3f(AXIS_LENGTH, 0, 0);
	glEnd();

	/* X axis cone */
	glPushMatrix();
	glTranslatef(AXIS_LENGTH, 0, 0);
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glutSolidCone(3, 10, 8, 8);
	glPopMatrix();

	/* Y axis - Green */
	glBegin(GL_LINES);
	glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
	glVertex3f(0, 0, 0);
	glVertex3f(0, AXIS_LENGTH, 0);
	glEnd();

	/* Y axis cone */
	glPushMatrix();
	glTranslatef(0, AXIS_LENGTH, 0);
	glRotatef(-90, 1, 0, 0);
	glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
	glutSolidCone(3, 10, 8, 8);
	glPopMatrix();

	/* Z axis - Blue */
	glBegin(GL_LINES);
	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, AXIS_LENGTH);
	glEnd();

	/* Z axis cone */
	glPushMatrix();
	glTranslatef(0, 0, AXIS_LENGTH);
	glRotatef(90, 0, 1, 0);
	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
	glutSolidCone(3, 10, 8, 8);
	glPopMatrix();

	/* Restore matrix */
	glPopMatrix();
}

void draw_grid(void) {
	int i;
	float size = GRID_SIZE;
	float step = GRID_STEP;

	glLineWidth(1.0f);
	glColor4f(0.3f, 0.3f, 0.5f, 0.3f);

	glBegin(GL_LINES);
	for (i = -size; i <= size; i += step) {
		glVertex3f(i, 0, -size);
		glVertex3f(i, 0, size);
		glVertex3f(-size, 0, i);
		glVertex3f(size, 0, i);
	}
	glEnd();
}

void draw_hypergraph(Hypergraph *hg) {
	int i, e;
	unsigned int src, tgt;
	ColorBGRA color;

	if (!hg || !hg->initialized)
		return;

	/* Draw visible edges */
	glLineWidth(1.0f);
	for (e = 0; e < hg->visible_edges; e++) {
		if (hg->edges[e].active) {
			src = hg->edges[e].source;
			tgt = hg->edges[e].target;

			glColor4ub(hg->edges[e].color.r, hg->edges[e].color.g,
					hg->edges[e].color.b,
					(uint8_t) (255 * (0.3f + hg->edges[e].luminescence * 0.7f)));

			glBegin(GL_LINES);
			glVertex3f(hg->nodes[src].x, hg->nodes[src].y, hg->nodes[src].z);
			glVertex3f(hg->nodes[tgt].x, hg->nodes[tgt].y, hg->nodes[tgt].z);
			glEnd();
		}
	}

	/* Draw visible nodes */
	glPointSize(4.0f);
	glBegin(GL_POINTS);
	for (i = 0; i < hg->visible_nodes; i++) {
		float act = hg->nodes[i].activation;
		color.r = (uint8_t) (hg->nodes[i].base_color.r * (1 - act)
				+ hg->nodes[i].active_color.r * act);
		color.g = (uint8_t) (hg->nodes[i].base_color.g * (1 - act)
				+ hg->nodes[i].active_color.g * act);
		color.b = (uint8_t) (hg->nodes[i].base_color.b * (1 - act)
				+ hg->nodes[i].active_color.b * act);
		color.a = 255;

		glColor4ub(color.r, color.g, color.b, color.a);
		glVertex3f(hg->nodes[i].x, hg->nodes[i].y, hg->nodes[i].z);
	}
	glEnd();
}

void draw_build_progress(Hypergraph *hg, OpenGLContext *gl, float fps) {
	char buffer[256];
	char *c;
	int ypos;
	const char *phase_names[] = { "ORIGIN", "BUILDING NODES", "BUILDING EDGES",
			"COMPLETE" };
	int phase_idx = (hg->build_phase < 3) ? hg->build_phase : 3;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, gl->width, 0, gl->height);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);

	/* Main HUD panel */
	glColor4f(0.0f, 0.0f, 0.0f, 0.7f);
	glBegin(GL_QUADS);
	glVertex2f(10, gl->height - 200);
	glVertex2f(500, gl->height - 200);
	glVertex2f(500, gl->height - 10);
	glVertex2f(10, gl->height - 10);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);

	/* Title */
	ypos = gl->height - 30;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "EVOX AI CORE v%s - 3D Interactive Preview", EVOX_VERSION);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);

	/* Stats */
	ypos -= 25;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "Phase: %s | Progress: %.1f%%", phase_names[phase_idx],
			hg->build_progress * 100);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "Nodes: %d/%d | Edges: %d/%d", hg->visible_nodes,
			hg->total_nodes, hg->visible_edges, hg->total_edges);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "Entropy: %.3f | FPS: %.1f", hg->global_entropy, fps);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 20;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "Rotation R: %.2f rad", gl->rotation_r);
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	/* Controls */
	ypos -= 25;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "CONTROLS:");
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 18;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "  Mouse Drag: Rotate View");
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 18;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "  R: Reset Camera | +/-: Zoom");
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	ypos -= 18;
	glRasterPos2f(20, ypos);
	sprintf(buffer, "  ESC: Exit");
	for (c = buffer; *c; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);

	/* Color legend */
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(gl->width - 150, 50);
	glVertex2f(gl->width - 100, 50);
	glVertex2f(gl->width - 100, 80);
	glVertex2f(gl->width - 150, 80);
	glEnd();

	glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(gl->width - 150, 90);
	glVertex2f(gl->width - 100, 90);
	glVertex2f(gl->width - 100, 120);
	glVertex2f(gl->width - 150, 120);
	glEnd();

	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);
	glVertex2f(gl->width - 150, 130);
	glVertex2f(gl->width - 100, 130);
	glVertex2f(gl->width - 100, 160);
	glVertex2f(gl->width - 150, 160);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);
	glRasterPos2f(gl->width - 90, 65);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'X');
	glRasterPos2f(gl->width - 90, 105);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Y');
	glRasterPos2f(gl->width - 90, 145);
	glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'Z');

	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

/*-----------------------------------------------------------------------------
 * MAIN APPLICATION
 *----------------------------------------------------------------------------*/

typedef struct {
	Hypergraph hypergraph;
	OpenGLContext gl;
	AudioContext audio;

	int running;
	uint64_t frame_count;
	float fps;
	uint64_t last_fps_time;

	pthread_t compute_thread;
	int initialized;
} EVOXApp;

static EVOXApp *g_app = NULL;
static int g_quit = 0;

void render_scene(EVOXApp *app) {
	float rad_angle, rad_elev, cam_x, cam_y, cam_z;

	if (!app || !app->initialized)
		return;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	/* Camera positioning */
	rad_angle = app->gl.camera_angle * (float) M_PI / 180.0f;
	rad_elev = app->gl.camera_elevation * (float) M_PI / 180.0f;

	cam_x = sinf(rad_angle) * cosf(rad_elev) * app->gl.camera_distance;
	cam_y = sinf(rad_elev) * app->gl.camera_distance;
	cam_z = cosf(rad_angle) * cosf(rad_elev) * app->gl.camera_distance;

	gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 1, 0);

	/* Update axes rotation */
	app->gl.rotation_r += ROTATION_SPEED;
	if (app->gl.rotation_r > 2.0f * (float) M_PI) {
		app->gl.rotation_r -= 2.0f * (float) M_PI;
	}

	/* Draw elements in order */
	draw_grid();
	draw_origin();
	draw_axes_rotating(app->gl.rotation_r);
	draw_hypergraph(&app->hypergraph);

	/* Draw HUD */
	draw_build_progress(&app->hypergraph, &app->gl, app->fps);

	SDL_GL_SwapWindow(app->gl.window);
	app->frame_count++;
}

void* compute_thread_func(void *arg) {
	EVOXApp *app = (EVOXApp*) arg;

	while (app->running && !g_quit) {
		update_hypergraph_build(&app->hypergraph);
		update_hypergraph_activity(&app->hypergraph);
		usleep(16000);
	}

	return NULL;
}

void handle_events(EVOXApp *app) {
	SDL_Event event;

	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_QUIT:
			app->running = 0;
			g_quit = 1;
			break;

		case SDL_KEYDOWN:
			switch (event.key.keysym.sym) {
			case SDLK_ESCAPE:
				app->running = 0;
				g_quit = 1;
				break;

			case SDLK_r:
				app->gl.camera_angle = 45.0f;
				app->gl.camera_elevation = 30.0f;
				app->gl.camera_distance = 400.0f;
				app->gl.rotation_r = 0.0f;
				break;

			case SDLK_EQUALS:
			case SDLK_PLUS:
				app->gl.camera_distance -= 20.0f;
				if (app->gl.camera_distance < 200.0f)
					app->gl.camera_distance = 200.0f;
				break;

			case SDLK_MINUS:
				app->gl.camera_distance += 20.0f;
				if (app->gl.camera_distance > 600.0f)
					app->gl.camera_distance = 600.0f;
				break;

			default:
				break;
			}
			break;

		case SDL_MOUSEBUTTONDOWN:
			if (event.button.button >= 1 && event.button.button <= 3) {
				app->gl.mouse_buttons[event.button.button - 1] = 1;
				app->gl.mouse_x = event.button.x;
				app->gl.mouse_y = event.button.y;
			}
			break;

		case SDL_MOUSEBUTTONUP:
			if (event.button.button >= 1 && event.button.button <= 3) {
				app->gl.mouse_buttons[event.button.button - 1] = 0;
			}
			break;

		case SDL_MOUSEMOTION:
			if (app->gl.mouse_buttons[0]) {
				app->gl.camera_angle += (event.motion.x - app->gl.mouse_x)
						* 0.3f;
				app->gl.camera_elevation += (event.motion.y - app->gl.mouse_y)
						* 0.3f;

				if (app->gl.camera_elevation > 85.0f)
					app->gl.camera_elevation = 85.0f;
				if (app->gl.camera_elevation < -85.0f)
					app->gl.camera_elevation = -85.0f;
			}
			app->gl.mouse_x = event.motion.x;
			app->gl.mouse_y = event.motion.y;
			break;

		case SDL_WINDOWEVENT:
			if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
				app->gl.width = event.window.data1;
				app->gl.height = event.window.data2;
				glViewport(0, 0, app->gl.width, app->gl.height);
			}
			break;

		default:
			break;
		}
	}
}

void init_app(EVOXApp *app) {
	SAFE_CHECK(app);

	printf("\n");
	printf("============================================================\n");
	printf("EVOX AI CORE v%s - %s\n", EVOX_VERSION, EVOX_CODENAME);
	printf("3D Interactive Preview - Starting from Origin (0,0,0)\n");
	printf("============================================================\n");

	memset(app, 0, sizeof(EVOXApp));

	srand(time(NULL));

	/* Initialize GLUT */
	{
		int fake_argc = 1;
		char *fake_argv[] = { "evox", NULL };
		glutInit(&fake_argc, fake_argv);
	}

	/* Initialize subsystems */
	init_audio(&app->audio);
	init_opengl(&app->gl, WINDOW_WIDTH, WINDOW_HEIGHT);
	init_hypergraph(&app->hypergraph, &app->audio);

	app->running = 1;
	app->frame_count = 0;
	app->fps = 0.0f;
	app->last_fps_time = get_timestamp_ms();
	app->initialized = 1;

	printf("============================================================\n");
	printf("SYSTEM INITIALIZED\n");
	printf("  OpenAL: %s\n", app->audio.initialized ? "Yes" : "No");
	printf("  OpenGL: Yes\n");
	printf("  Building: %d nodes, %d edges\n", app->hypergraph.total_nodes,
			app->hypergraph.total_edges);
	printf("============================================================\n\n");
}

void cleanup_app(EVOXApp *app) {
	if (!app)
		return;

	printf("\nShutting down...\n");

	app->running = 0;

	cleanup_audio(&app->audio);

	if (app->gl.initialized) {
		if (app->gl.gl_context)
			SDL_GL_DeleteContext(app->gl.gl_context);
		if (app->gl.window)
			SDL_DestroyWindow(app->gl.window);
		SDL_Quit();
	}

	printf("Cleanup complete\n");
}

/*-----------------------------------------------------------------------------
 * MAIN FUNCTION
 *----------------------------------------------------------------------------*/

int main(int argc, char **argv) {
	EVOXApp app;
	uint64_t now;
	uint64_t frame_counter = 0;
	uint64_t fps_timer = 0;

	(void) argc;
	(void) argv;

	signal(SIGINT, (void (*)(int)) cleanup_app);
	signal(SIGTERM, (void (*)(int)) cleanup_app);

	init_app(&app);
	g_app = &app;

	if (!app.gl.initialized) {
		fprintf(stderr, "Failed to initialize graphics\n");
		return 1;
	}

	pthread_create(&app.compute_thread, NULL, compute_thread_func, &app);

	fps_timer = get_timestamp_ms();

	while (app.running && !g_quit) {
		handle_events(&app);
		render_scene(&app);

		frame_counter++;
		now = get_timestamp_ms();
		if (now - fps_timer >= 1000) {
			app.fps = (float) frame_counter * 1000.0f / (now - fps_timer);
			frame_counter = 0;
			fps_timer = now;
		}

		SDL_Delay(1);
	}

	app.running = 0;
	pthread_join(app.compute_thread, NULL);
	cleanup_app(&app);

	printf("\n");
	printf("============================================================\n");
	printf("FINAL STATISTICS\n");
	printf("  Total frames: %llu\n", (unsigned long long) app.frame_count);
	printf("  Average FPS: %.1f\n", app.fps);
	printf("  Final build: %d/%d nodes, %d/%d edges\n",
			app.hypergraph.visible_nodes, app.hypergraph.total_nodes,
			app.hypergraph.visible_edges, app.hypergraph.total_edges);
	printf("============================================================\n");

	return 0;
}
