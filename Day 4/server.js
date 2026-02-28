require('dotenv').config();
const express = require('express');
const mysql = require('mysql2');
const session = require('express-session');
const bcrypt = require('bcryptjs');
const svgCaptcha = require('svg-captcha');
const multer = require('multer'); // Import Multer
const path = require('path');
const fs = require('fs');

const app = express();

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));
app.set('view engine', 'ejs');

// Session Config
app.use(session({
    secret: 'secret_key_change_this',
    resave: false,
    saveUninitialized: true
}));

// Database Connection
const db = mysql.createConnection({
    host: 'localhost',
    user: 'admin',
    password: 'Omi9321',
    database: 'user_portal'
});

db.connect((err) => {
    if (err) console.error('DB Error: ' + err.stack);
    else console.log('Connected to MariaDB...');
});

// --- MULTER CONFIGURATION (For File Uploads) ---
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const dir = './public/uploads';
        if (!fs.existsSync(dir)){
            fs.mkdirSync(dir, { recursive: true });
        }
        cb(null, dir);
    },
    filename: (req, file, cb) => {
        // Save as: userid-fieldname-timestamp.extension
        cb(null, req.session.user.id + '-' + file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage: storage });

// --- ROUTES ---

app.get('/', (req, res) => res.redirect('/login'));

app.get('/captcha', (req, res) => {
    const captcha = svgCaptcha.create();
    req.session.captcha = captcha.text;
    res.type('svg');
    res.status(200).send(captcha.data);
});

app.get('/login', (req, res) => res.render('login', { error: null }));

app.post('/login', (req, res) => {
    const { username, password, captcha } = req.body;
    if (req.session.captcha !== captcha) return res.render('login', { error: 'Invalid Captcha!' });

    db.query('SELECT * FROM users WHERE username = ?', [username], async (err, results) => {
        if (err) throw err;
        if (results.length === 0 || !(await bcrypt.compare(password, results[0].password))) {
            return res.render('login', { error: 'Invalid Credentials' });
        }
        req.session.user = results[0];
        res.redirect('/dashboard');
    });
});

app.get('/register', (req, res) => res.render('register', { error: null }));

app.post('/register', async (req, res) => {
    const { first_name, last_name, username, email, password, confirm_password } = req.body;
    if (password !== confirm_password) return res.render('register', { error: 'Passwords do not match!' });

    const hashedPassword = await bcrypt.hash(password, 10);
    const sql = 'INSERT INTO users (first_name, last_name, username, email, password) VALUES (?, ?, ?, ?, ?)';
    db.query(sql, [first_name, last_name, username, email, hashedPassword], (err) => {
        if (err) return res.render('register', { error: 'Username or Email already taken.' });
        res.redirect('/login');
    });
});

app.get('/dashboard', (req, res) => {
    if (!req.session.user) return res.redirect('/login');
    
    // Refresh user data (to get new file paths)
    db.query('SELECT * FROM users WHERE id = ?', [req.session.user.id], (err, users) => {
        if (err) throw err;
        req.session.user = users[0]; // Update session with latest DB data

        db.query('SELECT * FROM grades WHERE user_id = ?', [req.session.user.id], (err, grades) => {
            if (err) throw err;
            res.render('dashboard', { user: req.session.user, grades: grades });
        });
    });
});

// --- UPDATE PROFILE ROUTE (Handles Files + Text) ---
app.post('/update-profile', upload.fields([{ name: 'resume' }, { name: 'cover_letter' }]), (req, res) => {
    if (!req.session.user) return res.redirect('/login');

    const { username, email } = req.body;
    const userId = req.session.user.id;
    
    // Start with basic query
    let sql = 'UPDATE users SET username = ?, email = ?';
    let params = [username, email];

    // Check if Resume was uploaded
    if (req.files['resume']) {
        sql += ', resume = ?';
        params.push('/uploads/' + req.files['resume'][0].filename);
    }

    // Check if Cover Letter was uploaded
    if (req.files['cover_letter']) {
        sql += ', cover_letter = ?';
        params.push('/uploads/' + req.files['cover_letter'][0].filename);
    }

    sql += ' WHERE id = ?';
    params.push(userId);

    db.query(sql, params, (err) => {
        if (err) console.error(err);
        res.redirect('/dashboard');
    });
});

app.post('/reset-password', async (req, res) => {
    if (!req.session.user) return res.redirect('/login');
    const hashedPassword = await bcrypt.hash(req.body.new_password, 10);
    db.query('UPDATE users SET password = ? WHERE id = ?', [hashedPassword, req.session.user.id], (err) => {
        res.redirect('/dashboard');
    });
});

app.get('/logout', (req, res) => {
    req.session.destroy();
    res.redirect('/login');
});

const PORT = 3000;
app.listen(PORT, () => console.log('Server running on http://localhost:3000'));