/* Minimal WebGL2 + mat4 helpers.
 *
 * Deliberately dependency-free rather than pulling three.js from a CDN: this
 * page is served by a stdlib Python server on loopback and has to work with
 * no network at all. The scene is points and lines, which is a small enough
 * surface to hand-roll.
 */
'use strict';

const M4 = {
  create() { return new Float32Array(16); },

  identity(o) {
    o.fill(0); o[0] = o[5] = o[10] = o[15] = 1; return o;
  },

  perspective(o, fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    o.fill(0);
    o[0] = f / aspect; o[5] = f; o[11] = -1;
    o[10] = (far + near) * nf; o[14] = 2 * far * near * nf;
    return o;
  },

  lookAt(o, eye, center, up) {
    let z0 = eye[0] - center[0], z1 = eye[1] - center[1], z2 = eye[2] - center[2];
    let len = Math.hypot(z0, z1, z2) || 1;
    z0 /= len; z1 /= len; z2 /= len;

    let x0 = up[1] * z2 - up[2] * z1,
        x1 = up[2] * z0 - up[0] * z2,
        x2 = up[0] * z1 - up[1] * z0;
    len = Math.hypot(x0, x1, x2);
    if (!len) { x0 = 1; x1 = 0; x2 = 0; } else { x0 /= len; x1 /= len; x2 /= len; }

    const y0 = z1 * x2 - z2 * x1,
          y1 = z2 * x0 - z0 * x2,
          y2 = z0 * x1 - z1 * x0;

    o[0] = x0; o[1] = y0; o[2] = z0; o[3] = 0;
    o[4] = x1; o[5] = y1; o[6] = z1; o[7] = 0;
    o[8] = x2; o[9] = y2; o[10] = z2; o[11] = 0;
    o[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
    o[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
    o[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
    o[15] = 1;
    return o;
  },

  multiply(o, a, b) {
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      o[c * 4]     = a[0] * b0 + a[4] * b1 + a[8]  * b2 + a[12] * b3;
      o[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9]  * b2 + a[13] * b3;
      o[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      o[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return o;
  },

  /* Project a world point to normalised device coords (for hit-testing and
     HTML label placement). Returns null when behind the camera. */
  project(mvp, x, y, z) {
    const w = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (w <= 0) return null;
    return [
      (mvp[0] * x + mvp[4] * y + mvp[8]  * z + mvp[12]) / w,
      (mvp[1] * x + mvp[5] * y + mvp[9]  * z + mvp[13]) / w,
      w,
    ];
  },
};

const GLU = {
  shader(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('shader: ' + gl.getShaderInfoLog(s) + '\n' + src);
    }
    return s;
  },

  program(gl, vsSrc, fsSrc) {
    const p = gl.createProgram();
    gl.attachShader(p, GLU.shader(gl, gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, GLU.shader(gl, gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error('link: ' + gl.getProgramInfoLog(p));
    }
    // Cache locations up front; getUniformLocation per frame is a real cost.
    p.u = {}; p.a = {};
    const nu = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < nu; i++) {
      const name = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, '');
      p.u[name] = gl.getUniformLocation(p, name);
    }
    const na = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < na; i++) {
      const name = gl.getActiveAttrib(p, i).name;
      p.a[name] = gl.getAttribLocation(p, name);
    }
    return p;
  },

  buffer(gl, data, usage) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, usage || gl.STATIC_DRAW);
    return b;
  },

  attrib(gl, prog, name, buf, size, divisor) {
    const loc = prog.a[name];
    if (loc === undefined || loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    if (divisor !== undefined) gl.vertexAttribDivisor(loc, divisor);
  },
};
