const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

canvas.width = 200;
canvas.height = 200;

const camera = {
    position: vec3(0, 0, 30),
    forward: vec3(0, 0, -1),
    right: vec3(1, 0, 0),
    up: vec3(0, 1, 0),
    yaw: 0,
    pitch: 0,
    roll: 0
};

const origin0 = {x: 0, y: 0, z: 0};

const near = 0.01;

const keys = {
    w: false, a: false, s: false, d: false, arrowright: false, arrowleft: false, arrowup: false, arrowdown: false, spacebar: false, shift: false
}

const VERTEX_MODE = false;

const speed = 1;
const angleSpeed = 0.05;

const fov = 110*Math.PI/180;
const position = vec3(0,0,0); // whatever man i can make it more sophisticated later

// need to change the architexture but that can wait til later
const color_buffer = [];

// length canvas.width * canvas.height
const depth_buffer = [];
const pixel_buffer = []; // stores pixel color

// orientations: yaw, pitch, roll

const objects = []; // position: vec3, color: vec3, width: float, height: float, orientation: vec3() (yaw, pitch, roll)

// cube dimensions: [x_size, y_size, z_size]
// sphere dimensions: [radius, stacks, slices]

function makeObj(position, color, dim, orientation, type) {
    return {position, color, dim, orientation, type};
}

let face_vertices = [
    // front face
    0.5,  0.5,  0.5,
    0.5, -0.5,  0.5,
    -0.5, -0.5,  0.5,
    -0.5,  0.5,  0.5,
    // left face
    -0.5,  0.5,  0.5,
    -0.5, -0.5,  0.5,
    -0.5, -0.5, -0.5,
    -0.5,  0.5, -0.5,
    // back face -- front face but reverse z and reverse order
    -0.5,  0.5, -0.5,
    -0.5, -0.5, -0.5,
    0.5, -0.5, -0.5,
    0.5,  0.5, -0.5,
    // right face -- left face but reverse x and reverse order
    0.5,  0.5, -0.5,
    0.5, -0.5, -0.5,
    0.5, -0.5,  0.5,
    0.5,  0.5,  0.5,
    // top face
    0.5,  0.5, -0.5,
    0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
    -0.5,  0.5, -0.5,
    // bottom face -- bottom face but reverse y and reverse order
    -0.5,  -0.5, -0.5,
    -0.5,  -0.5,  0.5,
    0.5,  -0.5,  0.5,
    0.5,  -0.5, -0.5,
];

function cubeVerticesToTriangle(obj) {
    let position = obj.position;
    let x_size = obj.dim.x;
    let y_size = obj.dim.y;
    let z_size = obj.dim.z;
    let yaw = obj.orientation.x;
    let pitch = obj.orientation.y;
    let roll = obj.orientation.z;

    let vertex_buffer = [];
    for (let i = 0; i < face_vertices.length/12; i++) {
        // order is 0, 1, 2, 2, 3, 0

        let v0 = [x_size*face_vertices[i*12+0], y_size*face_vertices[i*12+1], z_size*face_vertices[i*12+2]];
        let v1 = [x_size*face_vertices[i*12+3], y_size*face_vertices[i*12+4], z_size*face_vertices[i*12+5]];
        let v2 = [x_size*face_vertices[i*12+6], y_size*face_vertices[i*12+7], z_size*face_vertices[i*12+8]];
        let v3 = [x_size*face_vertices[i*12+9], y_size*face_vertices[i*12+10], z_size*face_vertices[i*12+11]];

        let v0_vec = rotateVertex(vec3(v0[0], v0[1], v0[2]), origin0, yaw, pitch, roll);
        let v1_vec = rotateVertex(vec3(v1[0], v1[1], v1[2]), origin0, yaw, pitch, roll);
        let v2_vec = rotateVertex(vec3(v2[0], v2[1], v2[2]), origin0, yaw, pitch, roll);
        let v3_vec = rotateVertex(vec3(v3[0], v3[1], v3[2]), origin0, yaw, pitch, roll);

        v0_vec = vec3add(v0_vec, position);
        v1_vec = vec3add(v1_vec, position);
        v2_vec = vec3add(v2_vec, position);
        v3_vec = vec3add(v3_vec, position);

        v0 = [v0_vec.x, v0_vec.y, v0_vec.z];
        v1 = [v1_vec.x, v1_vec.y, v1_vec.z];
        v2 = [v2_vec.x, v2_vec.y, v2_vec.z];
        v3 = [v3_vec.x, v3_vec.y, v3_vec.z];

        vertex_buffer.push(
            ...v0, ...v1, ...v2, // triangle 1
            ...v2, ...v3, ...v0  // triangle 2
        );
    }
    return vertex_buffer;
}

function vec3(x=0, y=0, z=0) {
    return {x, y, z};
}

function vec3scale(k, v) {
    return vec3(k*v.x, k*v.y, k*v.z);
}

function vec3add(v1, v2) {
    return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

function vec3sub(v1, v2) {
    return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

function vec3cross(v1, v2) {
    return vec3(v1.y * v2.z - v2.y * v1.z, v1.z * v2.x - v2.z * v1.x, v1.x * v2.y - v2.x * v1.y);
}

function vec3dot(v1, v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

function vec3mul(v1, v2) { // for color vector math if necessary
    return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

function vec3dist(v1, v2) {
    return Math.hypot(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
}

function vec3mag(v) {
    return Math.hypot(v.x, v.y, v.z);
}

function normalize(v) {
    let mag = Math.hypot(v.x, v.y, v.z);
    return vec3(v.x/mag, v.y/mag, v.z/mag);
}

function vecToMat(v) {
    return {m:3, n:1, data: [v.x, v.y, v.z]};
}

/** returns `mat1` x `mat2` */
function matMult(mat1, mat2) {
    if (mat1.n !== mat2.m) throw new Error("multiplying incompatible matrix dimensions of " + mat1.n + " x " + mat2.m);

    let mat = {m: mat1.m, n: mat2.n, data: []};
    let m1 = mat1.m;
    let n1 = mat1.n; // n1 and m2 must be equal, dont need two
    // let m2 = mat2.m;
    let n2 = mat2.n;
    for (let m1_idx = 0; m1_idx < m1; m1_idx++) {
        for (let n2_idx = 0; n2_idx < n2; n2_idx++) {
            let sum = 0;
            for (let n1_idx = 0; n1_idx < n1; n1_idx++) {
                // matrix format = i * m + n
                // will be flipped reading mat2, mat2 stored as i * m + n, but i need data as i * n + m
                
                let mat1n = mat1.data[m1_idx * n1 + n1_idx];
                let mat2n = mat2.data[n1_idx * n2 + n2_idx];

                sum += mat1n * mat2n;
            }
            mat.data.push(sum);
        }
    }
    return mat;
}

function identity(size) {
    let mat = {m: size, n: size, data: []};
    for (let m = 0; m < size; m++) {
        for (let n = 0; n < size; n++) {
            mat.data.push(n===m ? 1 : 0);
        }
    }
    return mat;
}

// this is lowk useless lmfao
function compressMat(mat2D) {
    if (mat2D.length === 0) throw new Error("cannot compress empty matrix");
    let mat = {m: mat2D.length, n: mat2D[0].length, data: []};
    for (let m = 0; m < mat2D.length; m++) {
        for (let n = 0; n < mat2D[0].length; n++) {
            mat.data.push(mat2D[m][n]);
        }
    }
    return mat;
}

function mat3(data) {
    if (data.length === 0) throw new Error("cannot create matrix with empty data");
    return {m: 3, n: 3, data};
}

function matMultVec(mat, v) {
    let res = matMult(mat, vecToMat(v));
    return vec3(res.data[0], res.data[1], res.data[2]);
}

function rotateVertex(v, center, yaw, pitch, roll) {
    let pitchMat = mat3([
        1,               0,               0,
        0, Math.cos(pitch), -Math.sin(pitch),
        0, Math.sin(pitch),  Math.cos(pitch)
    ]);
    let rollMat = mat3([
        Math.cos(roll), -Math.sin(roll), 0,
        Math.sin(roll),  Math.cos(roll), 0,
        0,               0,              1
    ]);
    let yawMat = mat3([
        Math.cos(yaw),  0,  -Math.sin(yaw),
        0,              1,               0,
        Math.sin(yaw),  0,   Math.cos(yaw)
    ]);

    // order doesnt matter as long as they are all before vec
    let res = matMult(rollMat, pitchMat);
    res = matMult(res, yawMat);
    res = matMultVec(res, vec3sub(v, center));
    return vec3add(res, center);
}

/** must pass in projected vertices */
function getPointsInTriangle(v0, v1, v2) {
    // if (v0.z !== v1.z || v1.z !== v2.z || v0.z !== v2.z) throw new Error("pretty sure u want this to be flat. check ur work");

    // triangle bounding box. point in triangle if is on right side of all triangles

    let pixels = [];

    let minX = Math.floor(Math.min(v0.x, v1.x, v2.x));
    let maxX = Math.ceil(Math.max(v0.x, v1.x, v2.x));
    let minY = Math.floor(Math.min(v0.y, v1.y, v2.y));
    let maxY = Math.floor(Math.max(v0.y, v1.y, v2.y));

    // clamp to screen space
    minX = Math.max(minX, 0);
    minY = Math.max(minY, 0);
    maxX = Math.min(maxX, canvas.width-1);
    maxY = Math.min(maxY, canvas.height-1);

    // tweak if right/left are flipped
    function leftSideOfEdge(ax, ay, bx, by, px, py) {
        // 2d cross product
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    }

    for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
            // Sample pixel center
            let px = x + 0.5;
            let py = y + 0.5;

            let w0 = leftSideOfEdge(v1.x, v1.y, v2.x, v2.y, px, py);
            let w1 = leftSideOfEdge(v2.x, v2.y, v0.x, v0.y, px, py);
            let w2 = leftSideOfEdge(v0.x, v0.y, v1.x, v1.y, px, py);
            
            // if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
            if (w0 <= 0 && w1 <= 0 && w2 <= 0) {
                pixels.push(x, y);
            }
        }
    }
    return pixels;
}

function axisAngleMatrix(axis, angle) {
    axis = normalize(axis);
    let x = axis.x, y = axis.y, z = axis.z;
    let c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;

    return mat3([
        t*x*x + c,     t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z,   t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y,   t*y*z + s*x, t*z*z + c
    ]);
}

/** Camera orientation stored in camera object, this function adds to the orientation data */
function rotateCameraIncrement(yaw, pitch, roll) {
    camera.yaw += yaw;
    camera.pitch += pitch;
    camera.roll += roll;

    // yaw around world Y
    if (yaw !== 0) {
        let rotY = axisAngleMatrix(vec3(0,1,0), -yaw);
        camera.forward = matMultVec(rotY, camera.forward);
        camera.right   = matMultVec(rotY, camera.right);
        camera.up      = matMultVec(rotY, camera.up);
    }

    // pitch around local right
    if (pitch !== 0) {
        let rotX = axisAngleMatrix(camera.right, pitch);
        camera.forward = matMultVec(rotX, camera.forward);
        camera.up      = matMultVec(rotX, camera.up);
        // right unchanged
    }

    // roll around local forward
    if (roll !== 0) {
        let rotZ = axisAngleMatrix(camera.forward, roll);
        camera.right = matMultVec(rotZ, camera.right);
        camera.up    = matMultVec(rotZ, camera.up);
        // forward unchanged
    }

    // re-orthonormalize
    camera.forward = normalize(camera.forward);
    camera.right   = normalize(vec3cross(camera.forward, camera.up));
    camera.up      = vec3cross(camera.right, camera.forward);
}

function toCamera(v) {
    let rel = vec3sub(v, camera.position);
    return vec3(
        vec3dot(rel, camera.right),
        vec3dot(rel, camera.up),
        vec3dot(rel, camera.forward)
    );
}

// must be in camera space
function projectCam(v_cam) {
    if (v_cam.z <= near) return null; // extra guard
    const f = canvas.height / (2 * Math.tan(fov/2));
    return vec3(
        (v_cam.x/v_cam.z) * f + canvas.width/2,
        canvas.height/2 - (v_cam.y / v_cam.z) * f,
        v_cam.z
    );
}

function generateUVSphere(radius, stacks, slices) {
    let verts = []; // flat array of triangles: [x,y,z, x,y,z, ...]

    for (let i = 0; i < stacks; i++) {
        let theta1 = Math.PI * i / stacks;
        let theta2 = Math.PI * (i+1) / stacks;

        for (let j = 0; j < slices; j++) {
            let phi1 = 2 * Math.PI * j / slices;
            let phi2 = 2 * Math.PI * (j+1) / slices;

            // 4 corners of the patch
            let p1 = sphericalToCartesian(radius, theta1, phi1);
            let p2 = sphericalToCartesian(radius, theta2, phi1);
            let p3 = sphericalToCartesian(radius, theta2, phi2);
            let p4 = sphericalToCartesian(radius, theta1, phi2);

            // two triangles: (p1,p2,p3) and (p1,p3,p4)
            verts.push(...p1, ...p2, ...p3);
            verts.push(...p1, ...p3, ...p4);
        }
    }
    return verts;
}

function sphericalToCartesian(r, theta, phi) {
    return [
        r * Math.sin(theta) * Math.cos(phi),
        r * Math.cos(theta),
        r * Math.sin(theta) * Math.sin(phi),
    ];
}

function vertsToTriangles(verts) {
    let arr = [];

    for (let i = 0; i < verts.length/9; i++) {
        arr.push(
            verts[i*9+0], verts[i*9+1], verts[i*9+2],
            verts[i*9+3], verts[i*9+4], verts[i*9+5],
            verts[i*9+6], verts[i*9+7], verts[i*9+8],
            // mat: createMaterial(vec3(0, 0, 0.5))
        );
    }

    return arr;
}

function sphereVerticesToTriangles(obj) {
    let verts = generateUVSphere(obj.dim.x, obj.dim.y, obj.dim.z);
    let yaw = obj.orientation.x;
    let pitch = obj.orientation.y;
    let roll = obj.orientation.z;

    let vertex_buffer = [];
    for (let i = 0; i < verts.length / 3; i++) {
        let v = vec3(verts[i*3+0], verts[i*3+1], verts[i*3+2]);

        // rotate around origin
        v = rotateVertex(v, origin0, yaw, pitch, roll);

        // translate to obj.position
        v = vec3add(v, obj.position);

        vertex_buffer.push(v.x, v.y, v.z);
    }
    return vertex_buffer;
}

function flattenY(v) {
    return normalize(vec3(v.x,0,v.z));
}

function cameraControls() {
    let flatforward = flattenY(camera.forward);
    let flatRight = flattenY(camera.right);

    if (keys.w) camera.position = vec3add(camera.position, vec3scale(speed,flatforward));
    if (keys.a) camera.position = vec3add(camera.position, vec3scale(-speed,flatRight));
    if (keys.s) camera.position = vec3add(camera.position, vec3scale(-speed,flatforward));
    if (keys.d) camera.position = vec3add(camera.position, vec3scale(speed,flatRight));
    if (keys.arrowleft) rotateCameraIncrement(-angleSpeed,0,0);
    if (keys.arrowright) rotateCameraIncrement(angleSpeed,0,0);
    if (keys.arrowup) rotateCameraIncrement(0, angleSpeed, 0);
    if (keys.arrowdown) rotateCameraIncrement(0, -angleSpeed, 0);
    if (keys.spacebar) camera.position = vec3add(camera.position, vec3(0,speed,0));
    if (keys.shift) camera.position = vec3add(camera.position, vec3(0,-speed,0));
}

function start() {
    // vertices in clockwise cyclic order. idk why normal is flipped but it is what it is
    // normal = (v1 - v0) x (v2 - v0)

    window.addEventListener("keydown", e => {
        let key = e.key.toLowerCase();
        e.preventDefault();
        if (key === "w") keys.w = true;
        if (key === "a") keys.a = true;
        if (key === "s") keys.s = true;
        if (key === "d") keys.d = true;
        if (key === "arrowup") keys.arrowup = true;
        if (key === "arrowdown") keys.arrowdown = true;
        if (key === "arrowleft") keys.arrowleft = true;
        if (key === "arrowright") keys.arrowright = true;
        if (key === " ") keys.spacebar = true;
        if (key === "shift") keys.shift = true;
    });

    window.addEventListener("keyup", e => {
        let key = e.key.toLowerCase();
        e.preventDefault();
        if (key === "w") keys.w = false;
        if (key === "a") keys.a = false;
        if (key === "s") keys.s = false;
        if (key === "d") keys.d = false;
        if (key === "arrowup") keys.arrowup = false;
        if (key === "arrowdown") keys.arrowdown = false;
        if (key === "arrowleft") keys.arrowleft = false;
        if (key === "arrowright") keys.arrowright = false;
        if (key === " ") keys.spacebar = false;
        if (key === "shift") keys.shift = false;
    });

    // for right now random colors are gonna be just fine

    for (let x = -75; x < 75; x += 15) {
        for (let z = -75; z < 75; z += 15) {
            objects.push(makeObj(vec3(x, 0, z), origin0, vec3(10,10,10), vec3(Math.random() * 2 * Math.PI, Math.random() * 2 * Math.PI, Math.random() * 2 * Math.PI), "cube"));
        }
    }

    // sphere
    objects.push(makeObj(vec3(0,40,0), vec3(1,1,1), vec3(10,10,10), vec3(0,0,0), "sphere"));

    color_buffer.push(
        1,0,0,1,0,0,
        0,1,0,0,1,0,
        0,0,1,0,0,1,
        1,0,1,1,0,1,
        0,1,1,0,1,1,
        1,1,0,1,1,0
    );

    for (let i = 0; i < 10; i++) color_buffer.push(...color_buffer);

    for (let i = 0; i < canvas.width * canvas.height; i++) {
        depth_buffer.push(Infinity); // i = y * WIDTH + x
    }

    requestAnimationFrame(loop);
}

function loop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // let vertex_buffer = tris;

    // clear depth
    for (let i = 0; i < canvas.width * canvas.height; i++) {
        depth_buffer[i] = Infinity; // i = y * WIDTH + x
    }

    rotateCameraIncrement(0.00,0.00,0);

    cameraControls();

    for (const obj of objects) {
        let vertex_buffer = obj.type === "cube" ? cubeVerticesToTriangle(obj) : sphereVerticesToTriangles(obj);
        for (let i = 0; i < vertex_buffer.length/9; i++) {
            // reminder: normal = normalize((v2 - v0) x (v1 - v0))

            let v0 = vec3(vertex_buffer[i*9+0], vertex_buffer[i*9+1], vertex_buffer[i*9+2]);
            let v1 = vec3(vertex_buffer[i*9+3], vertex_buffer[i*9+4], vertex_buffer[i*9+5]);
            let v2 = vec3(vertex_buffer[i*9+6], vertex_buffer[i*9+7], vertex_buffer[i*9+8]);

            if (obj.type === "cube") {
                obj.orientation.x += 0.001;
                obj.orientation.y += 0.001;
                obj.orientation.z += 0.001;
            } else {
                obj.orientation = vec3add(obj.orientation, vec3(0.0001,0.0001,0.0001));
            }

            // world -> camera local space. use camera basis vectors. 
            let v0_cam = toCamera(v0);
            let v1_cam = toCamera(v1);
            let v2_cam = toCamera(v2);

            // backface culling must be done is camera space
            let normal_cam = normalize(vec3cross(vec3sub(v1_cam,v0_cam), vec3sub(v2_cam,v0_cam)));
            let view_dir = vec3scale(-1, v0_cam); // from triangle to camera
            if (vec3dot(normal_cam, view_dir) <= 0) continue;
            // if (normal_cam.z >= 0) continue; // normal facing in +z = we see back of triangle

            // clip near plane
            if (v0_cam.z <= near || v1_cam.z <= near || v2_cam.z <= near) continue;

            // project. also has world space depth
            const pv0 = projectCam(v0_cam);
            const pv1 = projectCam(v1_cam);
            const pv2 = projectCam(v2_cam);
            if (!pv0 || !pv1 || !pv2) continue; // extra near clipping guard
            
            if (VERTEX_MODE) {
                ctx.fillStyle = 'white';
                ctx.fillRect(pv0.x-2.5, pv0.y-2.5, 5, 5);
                ctx.fillRect(pv1.x-2.5, pv1.y-2.5, 5, 5);
                ctx.fillRect(pv2.x-2.5, pv2.y-2.5, 5, 5);
            }
            else {
                // triangle rasterizer
                let pixels = getPointsInTriangle(pv0, pv1, pv2);
                for (let j = 0; j < pixels.length/2; j++) {
                    let x = pixels[j*2+0];
                    let y = pixels[j*2+1];

                    let px = x + 0.5;
                    let py = y + 0.5;

                    // depth test
                    let depth = depth_buffer[y * canvas.width + x];

                    // barycentic? idk this is math that i dont feel like doing rn
                    // chatgpt made this
                    // use pv0/pv1/pv2 for weights
                    const x0 = pv0.x, y0 = pv0.y, z0 = pv0.z;
                    const x1 = pv1.x, y1 = pv1.y, z1 = pv1.z;
                    const x2 = pv2.x, y2 = pv2.y, z2 = pv2.z;

                    const denom = ((y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2));
                    if (denom === 0) continue; // degenerate

                    const alpha = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / denom;
                    const beta  = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / denom;
                    const gamma = 1 - alpha - beta;

                    const z = alpha*z0 + beta*z1 + gamma*z2; // this is your depth

                    if (z < depth) {
                        depth_buffer[y * canvas.width + x] = z;

                        let r = color_buffer[i*3+0] * 255;
                        let g = color_buffer[i*3+1] * 255;
                        let b = color_buffer[i*3+2] * 255;

                        // let r = Math.random() * 255;
                        // let g = Math.random() * 255;
                        // let b = Math.random() * 255;

                        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                        // ctx.fillStyle = 'white';
                        ctx.fillRect(x,y,1,1);
                    }
                }
            }
        }
    }

    requestAnimationFrame(loop);
}

start();
