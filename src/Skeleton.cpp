//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 20;

//---------------------------
struct Camera { // 3D camera
//---------------------------
    vec3 wEye, wLookat, wVup;   // extinsic
    float fov, asp, fp, bp;        // intrinsic
public:
    Camera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 10;
    } 
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0,   0,   0,   1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
                    0,                      1 / tan(fov / 2), 0,                      0,
                    0,                      0,                -(fp + bp) / (bp - fp), -1,
                    0,                      0,                -2 * fp*bp / (bp - fp),  0);
    }

    void Animate(float t) { }
};

//---------------------------
struct Material {
//---------------------------
    vec3 kd, ks, ka;
    float shininess;
};

//---------------------------
struct Light {
//---------------------------
    vec3 La, Le;
    vec4 wLightPos;

    void Animate(float t) {    }
};

//---------------------------
class RoomTexture : public Texture {
//---------------------------
public:
    RoomTexture(const int width = 600, const int height = 600) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 lightBlue(0.8f, 0.8f, 1, 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
            image[y * width + x] = lightBlue;
        }
        create(width, height, image, GL_NEAREST);
    }
};
//---------------------------
class AntibodyTexture : public Texture {
//---------------------------
public:
    AntibodyTexture(const int width = 600, const int height = 600) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 lightGreen(0.8f, 1, 0.8f, 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
            image[y * width + x] = lightGreen;
        }
        create(width, height, image, GL_NEAREST);
    }
};
//---------------------------
class VirusTexture : public Texture {
//---------------------------
public:
    VirusTexture(const int width = 0, const int height = 0) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 yellow(1, 1, 0, 1), darkRed(0.4f, 0, 0, 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
            image[y * width + x] = (x % 2) ? yellow : darkRed;
        }
        create(width, height, image, GL_NEAREST);
    }
};

//---------------------------
class VirusSpikeTexture : public Texture {
//---------------------------
public:
    VirusSpikeTexture(const int width = 600, const int height = 600) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 blue(0, 0, 1, 1), darkRed(0.4f, 0, 0, 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
            image[y * width + x] = blue ;
        }
        create(width, height, image, GL_NEAREST);
    }
};

//---------------------------
struct RenderState {
//---------------------------
    mat4               MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    Texture *          texture;
    vec3               wEye;
};

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

//---------------------------
class GouraudShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
        #version 330
        precision highp float;

        struct Light {
            vec3 La, Le;
            vec4 wLightPos;
        };
        
        struct Material {
            vec3 kd, ks, ka;
            float shininess;
        };

        uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
        uniform Light[8] lights;     // light source direction 
        uniform int   nLights;         // number of light sources
        uniform vec3  wEye;          // pos of eye
        uniform Material  material;  // diffuse, specular, ambient ref

        layout(location = 0) in vec3  vtxPos;            // pos in modeling space
        layout(location = 1) in vec3  vtxNorm;           // normal in modeling space

        out vec3 radiance;            // reflected radiance

        void main() {
            gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
            // radiance computation
            vec4 wPos = vec4(vtxPos, 1) * M;    
            vec3 V = normalize(wEye * wPos.w - wPos.xyz);
            vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
            if (dot(N, V) < 0) N = -N;    // prepare for one-sided surfaces like Mobius or Klein

            radiance = vec3(0, 0, 0);
            for(int i = 0; i < nLights; i++) {
                vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
                vec3 H = normalize(L + V);
                float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
                radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
            }
        }
    )";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
        #version 330
        precision highp float;

        in  vec3 radiance;      // interpolated radiance
        out vec4 fragmentColor; // output goes to frame buffer

        void main() {
            fragmentColor = vec4(radiance, 1);
        }
    )";
public:
    GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

//---------------------------
class PhongShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
        #version 330
        precision highp float;

        struct Light {
            vec3 La, Le;
            vec4 wLightPos;
        };

        uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
        uniform Light[8] lights;    // light sources 
        uniform int   nLights;
        uniform vec3  wEye;         // pos of eye

        layout(location = 0) in vec3  vtxPos;            // pos in modeling space
        layout(location = 1) in vec3  vtxNorm;           // normal in modeling space
        layout(location = 2) in vec2  vtxUV;

        out vec3 wNormal;            // normal in world space
        out vec3 wView;             // view in world space
        out vec3 wLight[8];            // light dir in world space
        out vec2 texcoord;

        void main() {
            gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
            // vectors for radiance computation
            vec4 wPos = vec4(vtxPos, 1) * M;
            for(int i = 0; i < nLights; i++) {
                wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
            }
            wView  = wEye * wPos.w - wPos.xyz;
            wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
            texcoord = vtxUV;
        }
    )";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
        #version 330
        precision highp float;

        struct Light {
            vec3 La, Le;
            vec4 wLightPos;
        };

        struct Material {
            vec3 kd, ks, ka;
            float shininess;
        };

        uniform Material material;
        uniform Light[8] lights;    // light sources 
        uniform int   nLights;
        uniform sampler2D diffuseTexture;

        in  vec3 wNormal;       // interpolated world sp normal
        in  vec3 wView;         // interpolated world sp view
        in  vec3 wLight[8];     // interpolated world sp illum dir
        in  vec2 texcoord;
        
        out vec4 fragmentColor; // output goes to frame buffer

        void main() {
            vec3 N = normalize(wNormal);
            vec3 V = normalize(wView); 
            if (dot(N, V) < 0) N = -N;    // prepare for one-sided surfaces like Mobius or Klein
            vec3 texColor = texture(diffuseTexture, texcoord).rgb;
            vec3 ka = material.ka * texColor;
            vec3 kd = material.kd * texColor;

            vec3 radiance = vec3(0, 0, 0);
            for(int i = 0; i < nLights; i++) {
                vec3 L = normalize(wLight[i]);
                vec3 H = normalize(L + V);
                float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
                // kd and ka are modulated by the texture
                radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
            }
            fragmentColor = vec4(radiance, 1);
        }
    )";
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

//---------------------------
class NPRShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
        #version 330
        precision highp float;

        uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
        uniform    vec4  wLightPos;
        uniform vec3  wEye;         // pos of eye

        layout(location = 0) in vec3  vtxPos;            // pos in modeling space
        layout(location = 1) in vec3  vtxNorm;           // normal in modeling space
        layout(location = 2) in vec2  vtxUV;

        out vec3 wNormal, wView, wLight;                // in world space
        out vec2 texcoord;

        void main() {
           gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
           vec4 wPos = vec4(vtxPos, 1) * M;
           wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
           wView  = wEye * wPos.w - wPos.xyz;
           wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
           texcoord = vtxUV;
        }
    )";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
        #version 330
        precision highp float;

        uniform sampler2D diffuseTexture;

        in  vec3 wNormal, wView, wLight;    // interpolated
        in  vec2 texcoord;
        out vec4 fragmentColor;                // output goes to frame buffer

        void main() {
           vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
           float y = (dot(N, L) > 0.5) ? 1 : 0.5;
           if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
           else                         fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
        }
    )";
public:
    NPRShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniform(state.lights[0].wLightPos, "wLightPos");
    }
};

//---------------------------
struct VertexData {
//---------------------------
    vec3 position, normal;
    vec2 texcoord;
};

//---------------------------
class Geometry {
//---------------------------
protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
    virtual float Animate(float tstart, float tend) { return 1.0f; }
};

//---------------------------
class ParamSurface : public Geometry {
//---------------------------
    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }

    virtual VertexData GenVertexData(float u, float v) = 0;

    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;    // vertices on the CPU
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
                vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
            }
        }
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};

//---------------------------
struct Clifford {
//---------------------------
    float f, d;
    Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
    Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
    Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
    Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
    Clifford operator/(Clifford r) {
        float l = r.f * r.f;
        return (*this) * Clifford(r.f / l, -r.d / l);
    }
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g)/Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

//---------------------------
class Virus : public ParamSurface {
//---------------------------
public:
    Virus() { create(); }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.position = vd.normal = vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       cosf(v * (float)M_PI));
        vd.texcoord = vec2(u, v);
        return vd;
    }
    float Animate(float tstart, float tend) { return 1.8f * tend; }
};

class Tractricoid : public ParamSurface {
//---------------------------
public:
    Tractricoid() { create(); }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        float height = 3;
        Clifford U(u * 2 * M_PI, 0), V(v * 2.0f, 1);
        Clifford X = cosf(v * M_PI * 2) / coshf(u * height);
        Clifford Y = sinf(v * M_PI * 2) / coshf(u * height);
        Clifford Z = u * height - tanhf(u * height);
        vd.position = vec3(X.f, Y.f, Z.f);

        // http://trecs.se/pseudosphere.php (normal vector)
        vd.normal = vec3(
            (powf(sinhf(u), 2) - 1.0f ) * cos(v),
            (powf(sinhf(u), 2) - 1.0f ) * sin(v),
            -1.0f * sinhf(u) * tanhf(u)
        );
        vd.texcoord = vec2(u, v);
        return vd;
    }
    float Animate(float tstart, float tend) { return 0.8f * tend; }
};

//---------------------------
class Antibody : public ParamSurface {
//---------------------------
public:
    Antibody() { create(); }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.position = vd.normal = vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                                       cosf(v * (float)M_PI));
        vd.texcoord = vec2(u, v);
        return vd;
    }
    float Animate(float tstart, float tend) { return 0.2f * tend; }
};

//---------------------------
class Room : public ParamSurface {
//---------------------------
public:
    Room() { create(); }

    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.position = vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                            sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
                            cosf(v * (float)M_PI));
        vd.normal = vec3(0,0,0) - vd.position;
        vd.texcoord = vec2(u, v);
        return vd;
    }
    float Animate(float tstart, float tend) { return 1.0f; }
};

//---------------------------
struct Object {
//---------------------------
    Shader *   shader;
    Material * material;
    Texture *  texture;
    Geometry * geometry;
    vec3 scale, translation, rotationAxis;
    float rotationAngle;
public:
    Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
        scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        texture = _texture;
        material = _material;
        geometry = _geometry;
    }
    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }

    virtual void Animate(float tstart, float tend) { 
        rotationAngle = geometry->Animate(tstart, tend);
    }
};

//---------------------------
class Scene {
//---------------------------
    std::vector<Object *> objects;
    Camera camera; // 3D camera
    std::vector<Light> lights;
public:
    void Build() {
        // Shaders
        Shader * phongShader = new PhongShader();

        // Materials
        Material * virusMaterial = new Material;
        virusMaterial->kd = vec3(0.6f, 0.4f, 0.2f);
        virusMaterial->ks = vec3(4, 4, 4);
        virusMaterial->ka = vec3(0.1f, 0.1f, 0.1f);
        virusMaterial->shininess = 100;

        Material * roomMaterial = new Material;
        roomMaterial->kd = vec3(0.8f, 0.6f, 0.4f);
        roomMaterial->ks = vec3(0.3f, 0.3f, 0.3f);
        roomMaterial->ka = vec3(0.2f, 0.2f, 0.2f);
        roomMaterial->shininess = 30;

        Material * antibodyMaterial = new Material;
        antibodyMaterial->kd = vec3(0.8f, 0.6f, 0.4f);
        antibodyMaterial->ks = vec3(0.3f, 0.3f, 0.3f);
        antibodyMaterial->ka = vec3(0.2f, 0.2f, 0.2f);
        antibodyMaterial->shininess = 30;

        // Textures
        Texture * roomTexture = new RoomTexture();
        Texture * virusTexture = new VirusTexture(30, 50);
        Texture * virusSpikeTexture = new VirusSpikeTexture(30, 50);
        Texture * antibodyTexture = new AntibodyTexture();

        // Geometries
        Geometry * room = new Room();
        Geometry * virus = new Virus();
        Geometry * tractricoid = new Tractricoid();
        Geometry * antibody = new Antibody();

        // Create objects by setting up their vertex data on the GPU
        Object * virusObject = new Object(phongShader, virusMaterial, virusTexture, virus);
        virusObject->translation = vec3(0, 0, 0);
        virusObject->rotationAxis = vec3(0, 1, 1);
        virusObject->scale = vec3(1.0f, 1.0f, 1.0f);
        objects.push_back(virusObject);

        Object * tractricoidObject = new Object(phongShader, virusMaterial, virusSpikeTexture, tractricoid);
        tractricoidObject->translation = vec3(-1, -1, 1);
        tractricoidObject->rotationAxis = vec3(0, 1, 1);
        tractricoidObject->scale = vec3(0.5f, 0.5f, 0.5f);
        objects.push_back(tractricoidObject);

        Object * antibodyObject = new Object(phongShader, antibodyMaterial, antibodyTexture, antibody);
        antibodyObject->translation = vec3(1, 1, 1);
        antibodyObject->rotationAxis = vec3(0, 1, 1);
        antibodyObject->scale = vec3(0.5f, 0.2f, 0.5f);
        objects.push_back(antibodyObject);

        Object * roomObject = new Object(phongShader, roomMaterial, roomTexture, room);
        roomObject->translation = vec3(0, 0, 0);
        roomObject->rotationAxis = vec3(0, 1, 1);
        roomObject->scale = vec3(3.5f, 3.5f, 3.5f);
        objects.push_back(roomObject);

        // Camera
        camera.wEye = vec3(0, 0, 4);
        camera.wLookat = vec3(0, 0, 0);
        camera.wVup = vec3(0, 1, 0);

        // Lights
        lights.resize(1);
        lights[0].wLightPos = vec4(0.09f, 0.07f, 0.06f, 0);    // ideal point -> directional light source
        lights[0].La = vec3(0.2f, 0.2f, 0.2f);
        lights[0].Le = vec3(3, 3, 3);
        
    }

    void Render() {
        RenderState state;
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.lights = lights;
        for (Object * obj : objects) obj->Draw(state);
    }

    void Animate(float tstart, float tend) {
        camera.Animate(tend);
        for (unsigned int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
        for (Object * obj : objects) obj->Animate(tstart, tend);
    }
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);                            // background color 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f; // dt is ”infinitesimal”
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.Animate(t, t + Dt);
    }
    glutPostRedisplay();
}
